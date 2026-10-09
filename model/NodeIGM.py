"""NodeIGM-inspired rumor graph classification with evidence-only pooling.

This is a graph-level adaptation, not a reproduction of the node-level code.
One gate is learned per undirected propagation relation. Training uses hard
forward masks with straight-through gradients, three environments obtained by
restoring environmental edges, and graph-level representation consistency.
Unlike the original inference-light design, inference retains the evidence gate.

Only nodes incident to a retained NON-SELF propagation edge enter readout.
This includes the source: if it becomes isolated it is excluded too. An entirely
edgeless evidence graph has a zero representation, hence classifier-bias logits.
Disconnected components containing edges remain eligible; root connectivity is
not imposed. Self-loops introduced inside GCNConv never rescue isolated nodes.

EIN interface: forward -> (log_probs, U, S, D), classification_loss(),
physics_loss() == 0, auxiliary_loss(), set_epoch(), init_optimizer().
Standalone training: loss = model.compute_loss(data).
"""

import math
from itertools import combinations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class NodeIGM(nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes, args, device=None):
        super().__init__()
        self.args = args
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hid_feats)
        self.num_layers = int(getattr(args, "n_layers_conv", 2))
        self.dropout = float(getattr(args, "dropout", 0.0))
        self.pool = str(getattr(args, "global_pool", "mean"))
        self.max_hop = int(getattr(args, "max_hop", 1))
        self.threshold = float(getattr(args, "nodeigm_edge_threshold", 0.5))
        self.keep_ratio = float(getattr(args, "nodeigm_keep_ratio", 0.7))
        # Disabled by default: source degree is not evidence in a rumor tree.
        self.degree_threshold = int(getattr(args, "nodeigm_degree_threshold", 0))
        self.env_weight = float(getattr(args, "nodeigm_env_weight", 0.1))
        self.variance_weight = float(getattr(args, "nodeigm_variance_weight", 1.0))
        self.keep_weight = float(getattr(args, "nodeigm_keep_weight", 0.01))
        self.warmup_epochs = int(getattr(args, "nodeigm_warmup_epochs", 0))
        self.mix_ratios = tuple(sorted(set(float(alpha) for alpha in
            getattr(args, "nodeigm_mix_ratios", (0.0, 0.1, 1.0)))))
        if min(int(in_feats), self.hidden_dim, self.num_classes,
               self.num_layers, self.max_hop) < 1:
            raise ValueError("NodeIGM dimensions and layer counts must be positive")
        if self.pool not in {"mean", "sum"}:
            raise ValueError("global_pool must be mean or sum")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.threshold < 1 or not 0 < self.keep_ratio < 1:
            raise ValueError("NodeIGM edge threshold and keep ratio must be in (0, 1)")
        if (len(self.mix_ratios) < 2 or self.mix_ratios[0] != 0.0
                or self.mix_ratios[-1] != 1.0
                or any(not 0 <= alpha <= 1 for alpha in self.mix_ratios)):
            raise ValueError("nodeigm_mix_ratios must include 0 and 1 and lie in [0, 1]")
        if self.degree_threshold < 0 or self.warmup_epochs < 0:
            raise ValueError("NodeIGM degree threshold and warmup must be nonnegative")
        if any(not math.isfinite(weight) or weight < 0 for weight in
               (self.env_weight, self.variance_weight, self.keep_weight)):
            raise ValueError("NodeIGM loss weights must be finite and nonnegative")

        self.convs = nn.ModuleList([
            GCNConv(int(in_feats) if layer == 0 else self.hidden_dim,
                    self.hidden_dim, cached=False)
            for layer in range(self.num_layers)
        ])
        self.edge_discriminator = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(), nn.Linear(4 * self.hidden_dim, 1), nn.Sigmoid(),
        )
        nn.init.constant_(self.edge_discriminator[-2].bias,
                          math.log(self.keep_ratio / (1 - self.keep_ratio)))
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        weights = getattr(args, "classification_class_weights", None)
        weights = torch.empty(0) if weights is None else torch.tensor(weights, dtype=torch.float32)
        if weights.numel() and (weights.shape != (self.num_classes,)
                or not torch.isfinite(weights).all() or (weights <= 0).any()):
            raise ValueError("classification_class_weights must have one positive weight per class")
        self.register_buffer("classification_class_weights", weights)
        self._epoch = 0
        self._last_auxiliary_loss = None
        self._last_diagnostics = {}
        # Like other models, placement is controlled by model.to(device).

    def _inputs(self, data):
        x = data.x.float()
        if x.ndim != 2 or x.size(0) == 0:
            raise ValueError("NodeIGM requires at least one node and 2-D node features")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if batch.shape != (x.size(0),) or batch.dtype != torch.long:
            raise ValueError("batch must contain one integer graph ID per node")
        num_graphs = int(batch.max().item()) + 1
        if batch.min() < 0 or torch.unique(batch).numel() != num_graphs:
            raise ValueError("batch graph IDs must be contiguous and start at zero")
        edge = getattr(data, "directed_edge_index", None)
        if edge is None:
            edge = data.edge_index
        if edge.ndim != 2 or edge.size(0) != 2 or edge.dtype != torch.long:
            raise ValueError("edge_index must be a [2, E] integer tensor")
        if edge.numel():
            if edge.min() < 0 or edge.max() >= x.size(0):
                raise ValueError("edge_index contains an out-of-range node")
            if (batch[edge[0]] != batch[edge[1]]).any():
                raise ValueError("propagation edges cannot connect different event graphs")
        edge = edge[:, edge[0] != edge[1]]
        # Tie both directions and duplicates to ONE gate. Self-loops are not
        # propagation evidence and are added separately by the encoder.
        edge = torch.unique(torch.sort(edge, dim=0).values, dim=1)
        return x, edge, batch, num_graphs

    @staticmethod
    def _bidirectional(edge, weight):
        return torch.cat((edge, edge.flip(0)), dim=1), weight.repeat(2)

    def _encode(self, x, edge, weight, dropout_masks=None):
        edge, weight = self._bidirectional(edge, weight)
        layers = []
        for layer, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge, edge_weight=weight))
            if dropout_masks is not None:
                x = x * dropout_masks[layer]
            layers.append(x)
        return layers

    def _edge_scores(self, x, edge):
        if edge.size(1) == 0:
            return x.new_empty(0)
        hidden = self._encode(x, edge, x.new_ones(edge.size(1)))[-1]
        row, col = edge
        # Endpoint-symmetric score: input node numbering must not determine
        # the importance of an undirected relation.
        pair = torch.cat((hidden[row], hidden[col]), dim=-1)
        reverse = torch.cat((hidden[col], hidden[row]), dim=-1)
        scores = 0.5 * (self.edge_discriminator(pair)
                        + self.edge_discriminator(reverse)).flatten()
        if self.degree_threshold:
            degree = torch.bincount(edge.flatten(), minlength=x.size(0))
            hubs = torch.maximum(degree[row], degree[col]) > self.degree_threshold
            scores = torch.where(hubs, torch.ones_like(scores), scores)
        return scores

    @staticmethod
    def _straight_through(hard, soft):
        # Parentheses preserve exact binary forward values (including zero).
        return hard.to(soft.dtype) + (soft - soft.detach())

    def _pool_weights(self, edge, hard_keep, soft_keep, num_nodes):
        degree = torch.bincount(edge[:, hard_keep].flatten(), minlength=num_nodes)
        active = degree > 0
        if not self.training:
            return active.to(soft_keep.dtype), active
        # Differentiable OR over incident edges. This supplies a gate gradient
        # even if a hard view has no surviving nodes. Forward readout still
        # excludes every isolated node exactly.
        eps = torch.finfo(soft_keep.dtype).eps
        log_absence = torch.log1p(-soft_keep.clamp(max=1 - eps))
        accumulated = soft_keep.new_zeros(num_nodes)
        accumulated.index_add_(0, edge[0], log_absence)
        accumulated.index_add_(0, edge[1], log_absence)
        soft_active = -torch.expm1(accumulated)
        return self._straight_through(active, soft_active), active

    def _readout(self, hidden, batch, node_weight, num_graphs):
        pooled = hidden.new_zeros(num_graphs, hidden.size(-1))
        pooled.index_add_(0, batch, hidden * node_weight[:, None])
        if self.pool == "mean":
            count = hidden.new_zeros(num_graphs)
            count.index_add_(0, batch, node_weight)
            pooled = pooled / count.clamp_min(1)[:, None]
        return pooled

    def _view(self, x, edge, batch, num_graphs, hard_keep, soft_keep, dropout_masks):
        weight = (self._straight_through(hard_keep, soft_keep) if self.training
                  else hard_keep.to(x.dtype))
        node_weight, active = self._pool_weights(edge, hard_keep, soft_keep, x.size(0))
        layers = self._encode(x, edge, weight, dropout_masks)
        graph_layers = [self._readout(hidden, batch, node_weight, num_graphs)
                        for hidden in layers]
        logits = self.classifier(graph_layers[-1])
        return F.log_softmax(logits, dim=-1), graph_layers + [logits], active

    def _restore_edges(self, edge, causal_mask, batch, num_graphs, alpha):
        if alpha == 0:
            return torch.zeros_like(causal_mask)
        if alpha == 1:
            return ~causal_mask
        restored = torch.zeros_like(causal_mask)
        edge_graph = batch[edge[0]]
        # Budget each event independently; never mix relations across events.
        for graph_id in range(num_graphs):
            candidates = ((edge_graph == graph_id) & ~causal_mask).nonzero().flatten()
            count = int(alpha * candidates.numel())
            if count:
                order = torch.randperm(candidates.numel(), device=edge.device)[:count]
                restored[candidates[order]] = True
        return restored

    def forward(self, data):
        x, edge, batch, num_graphs = self._inputs(data)
        scores = self._edge_scores(x, edge)
        causal_mask = scores > self.threshold
        dropout_masks = None
        if self.training and self.dropout:
            # Share randomness across environments: consistency should measure
            # edge changes, not independent dropout noise.
            dropout_masks = [
                (torch.rand(x.size(0), self.hidden_dim, device=x.device) >= self.dropout)
                .to(x.dtype) / (1 - self.dropout) for _ in self.convs
            ]
        output, evidence_reprs, active = self._view(
            x, edge, batch, num_graphs, causal_mask, scores, dropout_masks)
        env_loss = output.new_zeros(())
        keep_loss = output.new_zeros(())
        if self.training:
            representations = [evidence_reprs]
            for alpha in self.mix_ratios[1:]:
                restored = self._restore_edges(edge, causal_mask, batch, num_graphs, alpha)
                soft_keep = torch.where(restored, torch.ones_like(scores), scores)
                _, graph_reprs, _ = self._view(
                    x, edge, batch, num_graphs, causal_mask | restored,
                    soft_keep, dropout_masks)
                representations.append(graph_reprs)
            risks = torch.stack([
                torch.stack([F.mse_loss(a, b) for a, b in zip(left, right)]).mean()
                for left, right in combinations(representations, 2)
            ])
            env_loss = risks.mean() + self.variance_weight * risks.var(unbiased=False)
            if scores.numel():
                edge_graph = batch[edge[0]]
                sums = scores.new_zeros(num_graphs).index_add(0, edge_graph, scores)
                counts = torch.bincount(edge_graph, minlength=num_graphs)
                keep_loss = ((sums[counts > 0] / counts[counts > 0]
                              - self.keep_ratio).square()).mean()
        ramp = (min(1.0, (self._epoch + 1) / self.warmup_epochs)
                if self.warmup_epochs else 1.0)
        self._last_auxiliary_loss = (ramp * self.env_weight * env_loss
                                     + self.keep_weight * keep_loss)
        self._last_diagnostics = {
            "environment_loss": env_loss.detach(),
            "keep_loss": keep_loss.detach(),
            "edge_scores": scores.detach(),
            "evidence_edge_mask": causal_mask.detach(),
            "pool_mask": active.detach(),
            "graph_representation": evidence_reprs[-2].detach(),
        }
        placeholder = output.new_zeros(num_graphs, self.max_hop, 1)
        return output, placeholder, placeholder, placeholder

    @torch.no_grad()
    def extract_evidence_subgraph(self, data):
        """Return retained relations plus pool_mask in original node numbering.

        Nodes are not physically removed. pool_mask marks the exact nodes
        allowed in readout; false entries include an isolated source. Exposed
        propagation edges have both directions and contain no self-loops.
        """
        x, edge, batch, _ = self._inputs(data)
        scores = self._edge_scores(x, edge)
        keep = scores > self.threshold
        retained = edge[:, keep]
        pool_mask = torch.bincount(retained.flatten(), minlength=x.size(0)) > 0
        result = Data(x=x, edge_index=torch.cat((retained, retained.flip(0)), dim=1),
                      batch=batch, pool_mask=pool_mask,
                      original_node_id=torch.arange(x.size(0), device=x.device),
                      candidate_edge_index=edge, candidate_edge_scores=scores,
                      evidence_edge_mask=keep, num_nodes=x.size(0))
        return result

    def set_epoch(self, epoch):
        self._epoch = max(0, int(epoch))

    def classification_loss(self, output, target):
        weight = (self.classification_class_weights
                  if self.classification_class_weights.numel() else None)
        return F.nll_loss(output, target.view(-1).long(), weight=weight)

    def physics_loss(self, U, S, D, true_state):
        return self.classifier.weight.new_zeros(())

    def auxiliary_loss(self):
        return (self._last_auxiliary_loss if self._last_auxiliary_loss is not None
                else self.classifier.weight.new_zeros(()))

    def compute_loss(self, data):
        output, _, _, _ = self(data)
        return self.classification_loss(output, data.y) + self.auxiliary_loss()

    def get_diagnostics(self):
        return dict(self._last_diagnostics)

    def init_optimizer(self, args=None):
        args = self.args if args is None else args
        return torch.optim.Adam(self.parameters(), lr=getattr(args, "lr", 1e-3),
                                weight_decay=getattr(args, "weight_decay", 0.0))
