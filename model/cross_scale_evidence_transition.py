import torch
from torch import nn


class CrossScaleEvidenceStateTransition(nn.Module):
    """Model how local/global evidence states evolve over reply edges.

    The module turns node-aligned Change and Semantic-tree representations
    into four soft evidence states:

      0. verified change: local change and global relevance are both high;
      1. isolated change: local change is high but global relevance is low;
      2. global stability: global relevance is high without a large change;
      3. background: both signals are low.

    It then aggregates parent-to-child state transitions, optionally
    conditioned on support/deny edge mass and propagation phase.  The
    resulting graph pattern is injected into the two existing graph branches
    through small gated residuals.  There is no auxiliary objective, so the
    component can be evaluated as a self-contained architectural ablation.
    """

    state_names = (
        "verified_change",
        "isolated_change",
        "global_stability",
        "background",
    )

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_states = 4
        self.use_relation_condition = bool(
            getattr(args, "cest_use_relation_condition", True)
        )
        self.use_semantic_residual = bool(
            getattr(args, "cest_use_semantic_residual", True)
        )
        self.use_depth_phase = bool(
            getattr(args, "cest_use_depth_phase", False)
        )
        self.use_uncertainty_weight = bool(
            getattr(args, "cest_use_uncertainty_weight", False)
        )
        self.attention_scale = float(
            getattr(args, "cest_attention_scale", 1.0)
        )
        self.eps = max(1e-12, float(getattr(args, "cest_eps", 1e-6)))
        dropout = float(
            getattr(args, "cest_dropout", getattr(args, "dropout", 0.0))
        )

        self.change_score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.tree_score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

        relation_channels = 2 if self.use_relation_condition else 1
        phase_channels = 3 if self.use_depth_phase else 1
        transition_dim = (
            relation_channels
            * phase_channels
            * self.num_states
            * self.num_states
        )
        pattern_input_dim = self.num_states + transition_dim
        pattern_hidden_dim = max(
            1,
            int(getattr(args, "cest_hidden_dim", self.hidden_dim)),
        )
        self.pattern_encoder = nn.Sequential(
            nn.LayerNorm(pattern_input_dim),
            nn.Linear(pattern_input_dim, pattern_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pattern_hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        if self.use_semantic_residual:
            self.edge_semantic_encoder = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.edge_semantic_gate = nn.Sequential(
                nn.Linear(self.num_states * self.num_states + 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.edge_semantic_encoder = None
            self.edge_semantic_gate = None

        self.pattern_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.change_residual = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.tree_residual = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        gate_init = float(getattr(args, "cest_residual_gate_init", -2.0))
        self.change_gate_logit = nn.Parameter(torch.tensor(gate_init))
        self.tree_gate_logit = nn.Parameter(torch.tensor(gate_init))

    @staticmethod
    def _graph_count(batch):
        if batch.numel() == 0:
            return 0
        return int(batch.max().item()) + 1

    def _mean_by_graph(self, values, batch, num_graphs, weight=None):
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        if weight is None:
            weight = values.new_ones(values.size(0))
        weight = weight.to(dtype=values.dtype).view(-1)
        output = values.new_zeros((num_graphs, values.size(-1)))
        denominator = values.new_zeros((num_graphs, 1))
        if values.size(0) > 0:
            output.index_add_(0, batch, values * weight.unsqueeze(-1))
            denominator.index_add_(0, batch, weight.unsqueeze(-1))
        return output / denominator.clamp_min(self.eps)

    def _attention_node_score(
        self,
        attention_probability,
        valid_mask,
        expected_nodes,
    ):
        if attention_probability is None or valid_mask is None:
            raise ValueError(
                "CEST requires Semantic-tree pre-dropout attention and mask"
            )
        if attention_probability.dim() != 3 or valid_mask.dim() != 2:
            raise ValueError(
                "Semantic-tree attention/mask must have shapes [B,Q,N] and "
                "[B,N]"
            )
        if (
            attention_probability.size(0) != valid_mask.size(0)
            or attention_probability.size(2) != valid_mask.size(1)
        ):
            raise ValueError("Semantic-tree attention and mask do not align")
        mean_attention = attention_probability.mean(dim=1)
        node_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        relative_attention = mean_attention * node_count.to(
            dtype=mean_attention.dtype
        )
        dense_score = torch.log(relative_attention.clamp_min(self.eps))
        node_score = dense_score[valid_mask]
        if node_score.numel() != expected_nodes:
            raise ValueError(
                "Semantic-tree attention does not align with node features"
            )
        return node_score

    def _node_states(
        self,
        change_nodes,
        tree_nodes,
        attention_probability,
        valid_mask,
    ):
        if change_nodes.shape != tree_nodes.shape:
            raise ValueError(
                "CEST requires aligned Change and Semantic-tree node tensors"
            )
        if change_nodes.size(0) == 0:
            empty = change_nodes.new_empty((0, self.num_states))
            probability = change_nodes.new_empty((0,))
            return empty, probability, probability
        change_probability = torch.sigmoid(
            self.change_score(change_nodes).squeeze(-1)
        )
        attention_score = self._attention_node_score(
            attention_probability,
            valid_mask,
            change_nodes.size(0),
        )
        tree_logit = self.tree_score(tree_nodes).squeeze(-1)
        tree_probability = torch.sigmoid(
            tree_logit + self.attention_scale * attention_score
        )
        verified_change = change_probability * tree_probability
        isolated_change = change_probability * (1.0 - tree_probability)
        global_stability = (1.0 - change_probability) * tree_probability
        background = (1.0 - change_probability) * (
            1.0 - tree_probability
        )
        states = torch.stack(
            (
                verified_change,
                isolated_change,
                global_stability,
                background,
            ),
            dim=-1,
        )
        return states, change_probability, tree_probability

    def _edge_phase(self, depth, edge_index, batch, num_graphs):
        if not self.use_depth_phase:
            return edge_index.new_zeros(edge_index.size(1))
        if depth is None:
            raise ValueError("cest_use_depth_phase requires node depth")
        depth = depth.to(device=batch.device, dtype=torch.float32).view(-1)
        max_depth = depth.new_zeros(num_graphs)
        if hasattr(max_depth, "scatter_reduce_"):
            max_depth.scatter_reduce_(
                0,
                batch,
                depth,
                reduce="amax",
                include_self=True,
            )
        else:
            for graph_index in range(num_graphs):
                graph_depth = depth[batch == graph_index]
                if graph_depth.numel() > 0:
                    max_depth[graph_index] = graph_depth.max()
        child = edge_index[1]
        normalized_depth = depth[child] / max_depth[
            batch[child]
        ].clamp_min(1.0)
        return torch.clamp((normalized_depth * 3.0).long(), max=2)

    def _transition_profile(
        self,
        states,
        edge_index,
        support_weight,
        deny_weight,
        batch,
        depth,
        edge_uncertainty,
        num_graphs,
    ):
        relation_channels = 2 if self.use_relation_condition else 1
        phase_channels = 3 if self.use_depth_phase else 1
        flat_profile = states.new_zeros(
            (
                num_graphs * phase_channels * relation_channels,
                self.num_states,
                self.num_states,
            )
        )
        if edge_index.numel() == 0:
            return flat_profile.view(
                num_graphs,
                phase_channels,
                relation_channels,
                self.num_states,
                self.num_states,
            )

        parent, child = edge_index
        edge_batch = batch[parent]
        if not torch.equal(edge_batch, batch[child]):
            raise ValueError("CEST edges must not cross graph boundaries")
        outer = states[parent].unsqueeze(-1) * states[child].unsqueeze(-2)
        if self.use_relation_condition:
            relation_weight = torch.stack(
                (support_weight, deny_weight),
                dim=-1,
            )
        else:
            relation_weight = (support_weight + deny_weight).unsqueeze(-1)
        relation_weight = relation_weight.to(dtype=states.dtype).clamp_min(0)
        if self.use_uncertainty_weight:
            if edge_uncertainty is None:
                raise ValueError(
                    "cest_use_uncertainty_weight requires edge uncertainty"
                )
            reliability = 1.0 - edge_uncertainty.to(
                dtype=states.dtype
            ).clamp(0.0, 1.0)
            relation_weight = relation_weight * reliability.unsqueeze(-1)

        edge_phase = self._edge_phase(
            depth,
            edge_index,
            batch,
            num_graphs,
        )
        graph_denominator = states.new_zeros(num_graphs)
        relation_index = torch.arange(
            relation_channels,
            device=edge_index.device,
        )
        flat_index = (
            (
                edge_batch.unsqueeze(-1) * phase_channels
                + edge_phase.unsqueeze(-1)
            )
            * relation_channels
            + relation_index.unsqueeze(0)
        )
        flat_profile.index_add_(
            0,
            flat_index.reshape(-1),
            (
                outer.unsqueeze(1)
                * relation_weight.unsqueeze(-1).unsqueeze(-1)
            ).reshape(-1, self.num_states, self.num_states),
        )
        graph_denominator.index_add_(
            0,
            edge_batch,
            relation_weight.sum(dim=-1),
        )
        profile = flat_profile.view(
            num_graphs,
            phase_channels,
            relation_channels,
            self.num_states,
            self.num_states,
        )
        # Use one graph-level denominator instead of normalizing support and
        # deny independently.  This preserves their relative prevalence in
        # the transition fingerprint while remaining invariant to graph size.
        return profile / graph_denominator.clamp_min(self.eps).view(
            -1, 1, 1, 1, 1
        )

    def _semantic_residual(
        self,
        change_nodes,
        tree_nodes,
        states,
        edge_index,
        support_weight,
        deny_weight,
        batch,
        edge_uncertainty,
        num_graphs,
    ):
        if not self.use_semantic_residual:
            return change_nodes.new_zeros((num_graphs, self.hidden_dim))
        if edge_index.numel() == 0:
            return change_nodes.new_zeros((num_graphs, self.hidden_dim))
        parent, child = edge_index
        edge_semantics = self.edge_semantic_encoder(
            torch.cat(
                (
                    (change_nodes[parent] - change_nodes[child]).abs(),
                    (tree_nodes[parent] - tree_nodes[child]).abs(),
                ),
                dim=-1,
            )
        )
        outer = states[parent].unsqueeze(-1) * states[child].unsqueeze(-2)
        relation = torch.stack(
            (support_weight, deny_weight),
            dim=-1,
        ).to(dtype=change_nodes.dtype)
        semantic_gate = self.edge_semantic_gate(
            torch.cat((outer.flatten(1), relation), dim=-1)
        ).squeeze(-1)
        edge_mass = relation.sum(dim=-1).clamp_min(0.0)
        if self.use_uncertainty_weight and edge_uncertainty is not None:
            edge_mass = edge_mass * (
                1.0
                - edge_uncertainty.to(dtype=edge_mass.dtype).clamp(0.0, 1.0)
            )
        return self._mean_by_graph(
            edge_semantics,
            batch[parent],
            num_graphs,
            weight=semantic_gate * edge_mass,
        )

    def forward(
        self,
        change_nodes,
        tree_nodes,
        tree_attention_probability,
        tree_valid_mask,
        edge_index,
        support_weight,
        deny_weight,
        batch,
        change_graph,
        tree_graph,
        depth=None,
        edge_uncertainty=None,
    ):
        if change_graph is None or tree_graph is None:
            raise ValueError("CEST requires Change and Semantic-tree graphs")
        num_graphs = self._graph_count(batch)
        if change_graph.size(0) != num_graphs or tree_graph.size(0) != num_graphs:
            raise ValueError("CEST graph representations do not match batch")

        states, change_probability, tree_probability = self._node_states(
            change_nodes,
            tree_nodes,
            tree_attention_probability,
            tree_valid_mask,
        )
        node_profile = self._mean_by_graph(
            states,
            batch,
            num_graphs,
        )
        transition_profile = self._transition_profile(
            states,
            edge_index,
            support_weight,
            deny_weight,
            batch,
            depth,
            edge_uncertainty,
            num_graphs,
        )
        structural_pattern = self.pattern_encoder(
            torch.cat(
                (node_profile, transition_profile.flatten(1)),
                dim=-1,
            )
        )
        semantic_residual = self._semantic_residual(
            change_nodes,
            tree_nodes,
            states,
            edge_index,
            support_weight,
            deny_weight,
            batch,
            edge_uncertainty,
            num_graphs,
        )
        pattern_graph = self.pattern_output(
            structural_pattern + semantic_residual
        )

        change_gate = torch.sigmoid(self.change_gate_logit)
        tree_gate = torch.sigmoid(self.tree_gate_logit)
        refined_change_graph = (
            change_graph
            + change_gate * self.change_residual(pattern_graph)
        )
        refined_tree_graph = (
            tree_graph
            + tree_gate * self.tree_residual(pattern_graph)
        )
        return {
            "refined_change_graph": refined_change_graph,
            "refined_tree_graph": refined_tree_graph,
            "pattern_graph": pattern_graph,
            "node_states": states,
            "node_profile": node_profile,
            "transition_profile": transition_profile,
            "change_probability": change_probability,
            "tree_probability": tree_probability,
            "semantic_residual": semantic_residual,
            "change_gate": change_gate,
            "tree_gate": tree_gate,
        }
