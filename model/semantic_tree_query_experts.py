import torch
import torch.nn.functional as F
from torch import nn


class LowRankSemanticTreeQueryExperts(nn.Module):
    """Low-rank, event-adaptive diagnostic query experts.

    The module does not perform node cross-attention itself. It prepares a
    bank of event-conditioned queries and, after the caller has applied the
    shared Semantic-tree attention layers, discovers balanced expert
    responsibilities and fuses specialist outputs as a residual correction to
    the existing generalist query.
    """

    def __init__(self, hidden_dim, num_classes, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_experts = max(
            1,
            int(getattr(args, "semantic_tree_query_expert_num", 4)),
        )
        self.top_k = min(
            self.num_experts,
            max(
                1,
                int(getattr(args, "semantic_tree_query_expert_topk", 2)),
            ),
        )
        self.basis_rank = min(
            self.hidden_dim,
            max(
                1,
                int(
                    getattr(
                        args,
                        "semantic_tree_query_expert_basis_rank",
                        min(8, self.hidden_dim),
                    )
                ),
            ),
        )
        self.adapter_rank = max(
            1,
            int(
                getattr(
                    args,
                    "semantic_tree_query_expert_adapter_rank",
                    min(8, self.hidden_dim),
                )
            ),
        )
        self.router_temperature = max(
            1e-6,
            float(
                getattr(
                    args,
                    "semantic_tree_query_expert_router_temperature",
                    1.0,
                )
            ),
        )
        self.responsibility_temperature = max(
            1e-6,
            float(
                getattr(
                    args,
                    "semantic_tree_query_expert_responsibility_temperature",
                    0.5,
                )
            ),
        )
        self.competence_weight = max(
            0.0,
            float(
                getattr(
                    args,
                    "semantic_tree_query_expert_competence_weight",
                    1.0,
                )
            ),
        )
        self.warmup_epochs = max(
            0,
            int(
                getattr(
                    args,
                    "semantic_tree_query_expert_warmup_epochs",
                    5,
                )
            ),
        )
        self.sinkhorn_iterations = max(
            1,
            int(
                getattr(
                    args,
                    "semantic_tree_query_expert_sinkhorn_iterations",
                    3,
                )
            ),
        )
        self.counterfactual_margin = max(
            0.0,
            float(
                getattr(
                    args,
                    "semantic_tree_query_expert_counterfactual_margin",
                    0.1,
                )
            ),
        )
        self.detach_responsibility = bool(
            getattr(
                args,
                "semantic_tree_query_expert_detach_responsibility",
                True,
            )
        )
        residual_gate_init = float(
            getattr(
                args,
                "semantic_tree_query_expert_residual_gate_init",
                -4.0,
            )
        )
        self.eps = 1e-6
        self.register_buffer(
            "_current_epoch",
            torch.zeros((), dtype=torch.long),
        )

        event_dim = self.hidden_dim * 2
        self.event_norm = nn.LayerNorm(event_dim)
        self.event_adapter = nn.Sequential(
            nn.Linear(event_dim, self.adapter_rank),
            nn.GELU(),
        )
        self.event_coefficient_delta = nn.Linear(
            self.adapter_rank,
            self.num_experts * self.basis_rank,
        )
        nn.init.zeros_(self.event_coefficient_delta.weight)
        nn.init.zeros_(self.event_coefficient_delta.bias)

        self.query_basis = nn.Parameter(
            torch.empty(self.basis_rank, self.hidden_dim)
        )
        self.expert_coefficients = nn.Parameter(
            torch.empty(self.num_experts, self.basis_rank)
        )
        nn.init.normal_(
            self.query_basis,
            std=self.hidden_dim ** -0.5,
        )
        nn.init.normal_(self.expert_coefficients, std=0.02)
        self.query_norm = nn.LayerNorm(self.hidden_dim)

        self.router = nn.Linear(event_dim, self.num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.residual_gate = nn.Linear(event_dim, 1)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, residual_gate_init)

        if num_classes is None:
            raise ValueError(
                "semantic-tree query experts require num_classes"
            )
        self.expert_classifier = nn.Linear(
            self.hidden_dim,
            int(num_classes),
        )

        self.last_event_context = None
        self.last_queries = None
        self.last_router_logits = None
        self.last_router_probability = None
        self.last_route_weight = None
        self.last_responsibility = None
        self.last_residual_gate = None
        self.last_attention = None
        self.last_graphs = None
        self.last_classification_loss = None
        self.last_routing_loss = None
        self.last_diversity_loss = None
        self.last_counterfactual_loss = None

    @property
    def current_epoch(self):
        return int(self._current_epoch.item())

    def set_epoch(self, epoch):
        self._current_epoch.fill_(max(0, int(epoch)))

    def _event_context(self, original_dense, valid_mask):
        valid_weight = valid_mask.to(
            dtype=original_dense.dtype
        ).unsqueeze(-1)
        graph_mean = (original_dense * valid_weight).sum(dim=1)
        graph_mean = graph_mean / valid_weight.sum(dim=1).clamp_min(1.0)
        root = original_dense[:, 0]
        return self.event_norm(torch.cat((graph_mean, root), dim=-1))

    def _route_weights(self, router_logits):
        dense_probability = F.softmax(
            router_logits / self.router_temperature,
            dim=-1,
        )
        if self.training and self.current_epoch < self.warmup_epochs:
            route_weight = torch.full_like(
                dense_probability,
                1.0 / self.num_experts,
            )
            return dense_probability, route_weight
        if self.top_k >= self.num_experts:
            return dense_probability, dense_probability

        top_index = router_logits.topk(self.top_k, dim=-1).indices
        selected = torch.zeros_like(
            router_logits,
            dtype=torch.bool,
        )
        selected.scatter_(1, top_index, True)
        sparse_logits = (
            router_logits / self.router_temperature
        ).masked_fill(~selected, -1e9)
        return dense_probability, F.softmax(sparse_logits, dim=-1)

    def prepare_queries(self, general_query, original_dense, valid_mask):
        if general_query.size(1) != 1:
            raise ValueError(
                "semantic-tree query experts require exactly one generalist "
                "query"
            )
        event_context = self._event_context(original_dense, valid_mask)
        coefficient_delta = self.event_coefficient_delta(
            self.event_adapter(event_context)
        ).view(-1, self.num_experts, self.basis_rank)
        coefficients = (
            self.expert_coefficients.unsqueeze(0) + coefficient_delta
        )
        query_offset = torch.matmul(coefficients, self.query_basis)
        expert_queries = self.query_norm(
            general_query.expand(-1, self.num_experts, -1)
            + query_offset
        )
        router_logits = self.router(event_context)
        router_probability, route_weight = self._route_weights(router_logits)
        return {
            "event_context": event_context,
            "expert_queries": expert_queries,
            "router_logits": router_logits,
            "router_probability": router_probability,
            "route_weight": route_weight,
        }

    def _sinkhorn(self, score):
        if score.numel() == 0:
            return F.softmax(score, dim=-1)
        stabilized = score / self.responsibility_temperature
        stabilized = stabilized - stabilized.max()
        assignment = stabilized.exp().t()
        assignment = assignment / assignment.sum().clamp_min(self.eps)
        num_experts, num_samples = assignment.shape
        for _ in range(self.sinkhorn_iterations):
            assignment = assignment / assignment.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(self.eps)
            assignment = assignment / float(num_experts)
            assignment = assignment / assignment.sum(
                dim=0,
                keepdim=True,
            ).clamp_min(self.eps)
            assignment = assignment / float(num_samples)
        return (assignment * float(num_samples)).t()

    def _class_balanced_responsibility(self, score, target):
        responsibility = torch.zeros_like(score)
        for class_id in target.unique(sorted=True):
            class_mask = target.eq(class_id)
            responsibility[class_mask] = self._sinkhorn(score[class_mask])
        return responsibility

    def _attention_diversity(self, attention, valid_mask):
        zero = attention.new_zeros(())
        if self.num_experts < 2:
            return zero
        masked = attention * valid_mask.unsqueeze(1).to(attention.dtype)
        normalized = F.normalize(masked, p=2, dim=-1, eps=self.eps)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        identity = torch.eye(
            self.num_experts,
            dtype=torch.bool,
            device=attention.device,
        ).unsqueeze(0)
        return similarity.masked_select(~identity).pow(2).mean()

    def fuse(
        self,
        general_graph,
        expert_graphs,
        general_query,
        expert_queries,
        general_context,
        expert_contexts,
        general_attention,
        expert_attention,
        general_attention_probability,
        expert_attention_probability,
        valid_mask,
        prepared,
        target=None,
    ):
        route_weight = prepared["route_weight"]
        event_context = prepared["event_context"]
        residual_gate = torch.sigmoid(self.residual_gate(event_context))
        expert_mix = (
            route_weight.unsqueeze(-1) * expert_graphs
        ).sum(dim=1)
        graph_hidden = (
            general_graph
            + residual_gate * (expert_mix - general_graph)
        )

        def residual_mix(general, experts):
            mixture = (route_weight.unsqueeze(-1) * experts).sum(dim=1)
            return general + residual_gate * (mixture - general)

        query = residual_mix(
            general_query.squeeze(1),
            expert_queries,
        ).unsqueeze(1)
        context = residual_mix(
            general_context.squeeze(1),
            expert_contexts,
        ).unsqueeze(1)
        attention = residual_mix(
            general_attention.squeeze(1),
            expert_attention,
        ).unsqueeze(1)
        attention_probability = residual_mix(
            general_attention_probability.squeeze(1),
            expert_attention_probability,
        ).unsqueeze(1)

        diversity_loss = self._attention_diversity(
            expert_attention_probability,
            valid_mask,
        )
        classification_loss = None
        routing_loss = None
        counterfactual_loss = None
        responsibility = None
        expert_logits = self.expert_classifier(expert_graphs)
        if target is not None:
            target = target.view(-1).long()
            if target.numel() != expert_graphs.size(0):
                raise ValueError(
                    "semantic-tree expert target size does not match batch"
                )
            expert_log_probability = F.log_softmax(expert_logits, dim=-1)
            correct_log_probability = expert_log_probability.gather(
                2,
                target.view(-1, 1, 1).expand(
                    -1,
                    self.num_experts,
                    1,
                ),
            ).squeeze(-1)
            responsibility_score = (
                prepared["router_logits"]
                + self.competence_weight * correct_log_probability
            )
            if self.detach_responsibility:
                responsibility_score = responsibility_score.detach()
            responsibility = self._class_balanced_responsibility(
                responsibility_score,
                target,
            )
            if self.detach_responsibility:
                responsibility = responsibility.detach()

            classification_loss = -(
                responsibility * correct_log_probability
            ).sum(dim=-1).mean()
            routing_loss = (
                responsibility
                * (
                    responsibility.clamp_min(self.eps).log()
                    - prepared["router_probability"]
                    .clamp_min(self.eps)
                    .log()
                )
            ).sum(dim=-1).mean()

            zero = graph_hidden.new_zeros(())
            if self.training and self.current_epoch < self.warmup_epochs:
                counterfactual_loss = zero
            else:
                weighted_sum = (
                    route_weight.unsqueeze(-1) * expert_graphs
                ).sum(dim=1)
                remaining_mass = 1.0 - route_weight
                without_sum = (
                    weighted_sum.unsqueeze(1)
                    - route_weight.unsqueeze(-1) * expert_graphs
                )
                without_mix = without_sum / remaining_mass.unsqueeze(
                    -1
                ).clamp_min(self.eps)
                use_general = remaining_mass <= self.eps
                without_mix = torch.where(
                    use_general.unsqueeze(-1),
                    general_graph.unsqueeze(1).expand_as(without_mix),
                    without_mix,
                )
                without_graph = (
                    general_graph.unsqueeze(1)
                    + residual_gate.unsqueeze(1)
                    * (without_mix - general_graph.unsqueeze(1))
                )
                full_logits = self.expert_classifier(graph_hidden)
                without_logits = self.expert_classifier(without_graph)
                full_correct = full_logits.gather(
                    1,
                    target.unsqueeze(1),
                ).squeeze(1)
                without_correct = without_logits.gather(
                    2,
                    target.view(-1, 1, 1).expand(
                        -1,
                        self.num_experts,
                        1,
                    ),
                ).squeeze(-1)
                contribution = full_correct.unsqueeze(1) - without_correct
                selected = route_weight.gt(0.0).to(route_weight.dtype)
                counterfactual_weight = responsibility * selected
                counterfactual_loss = (
                    counterfactual_weight
                    * F.relu(self.counterfactual_margin - contribution)
                ).sum() / counterfactual_weight.sum().clamp_min(1.0)

        self.last_event_context = event_context
        self.last_queries = expert_queries
        self.last_router_logits = prepared["router_logits"]
        self.last_router_probability = prepared["router_probability"]
        self.last_route_weight = route_weight
        self.last_responsibility = responsibility
        self.last_residual_gate = residual_gate
        self.last_attention = expert_attention_probability
        self.last_graphs = expert_graphs
        self.last_classification_loss = classification_loss
        self.last_routing_loss = routing_loss
        self.last_diversity_loss = diversity_loss
        self.last_counterfactual_loss = counterfactual_loss
        return {
            "graph_hidden": graph_hidden,
            "query": query,
            "context": context,
            "attention": attention,
            "attention_probability": attention_probability,
        }
