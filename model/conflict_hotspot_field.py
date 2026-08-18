"""Conflict-hotspot diffusion for local-to-global semantic coordination."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_max_pool, global_min_pool
from torch_geometric.utils import to_dense_batch


class ConflictHotspotField(nn.Module):
    """Diffuse node-level change intensity over a propagation tree.

    The module is deliberately not a third classification branch.  It emits
    (1) a positive multiplier for Change pooling and (2) a node bias for the
    Semantic-tree readout.  Both consumers are independently switchable.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.steps = max(
            0,
            int(getattr(args, "conflict_hotspot_diffusion_steps", 2)),
        )
        self.alpha = min(
            max(float(getattr(args, "conflict_hotspot_diffusion_alpha", 0.35)), 0.0),
            1.0,
        )
        self.direction = str(
            getattr(args, "conflict_hotspot_diffusion_direction", "undirected")
        ).strip().lower()
        direction_aliases = {
            "both": "undirected",
            "bidirectional": "undirected",
            "forward": "top_down",
            "backward": "bottom_up",
        }
        self.direction = direction_aliases.get(self.direction, self.direction)
        if self.direction not in {"undirected", "top_down", "bottom_up"}:
            raise ValueError(
                "conflict_hotspot_diffusion_direction must be one of "
                "['bottom_up', 'top_down', 'undirected'], got {}".format(
                    self.direction
                )
            )

        self.use_change_pooling = bool(
            getattr(args, "conflict_hotspot_use_change_pooling", True)
        )
        self.change_pool_scale = max(
            0.0,
            float(getattr(args, "conflict_hotspot_change_pool_scale", 1.0)),
        )
        self.use_semantic_tree_bias = bool(
            getattr(args, "conflict_hotspot_use_semantic_tree_bias", True)
        )
        self.semantic_tree_bias_scale = max(
            0.0,
            float(
                getattr(args, "conflict_hotspot_semantic_tree_bias_scale", 1.0)
            ),
        )
        self.coverage_temperature = max(
            1e-6,
            float(getattr(args, "conflict_hotspot_coverage_temperature", 0.5)),
        )
        self.lambda_coverage = max(
            0.0,
            float(getattr(args, "lambda_conflict_hotspot_coverage_aux", 0.0)),
        )
        self.detach_coverage_teacher = bool(
            getattr(args, "conflict_hotspot_coverage_detach", True)
        )
        dropout = min(
            max(float(getattr(args, "conflict_hotspot_dropout", 0.0)), 0.0),
            1.0,
        )
        self.intensity = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.eps = 1e-6
        self.last_local_intensity = None
        self.last_field_intensity = None
        self.last_normalized_field = None
        self.last_pool_multiplier = None
        self.last_attention_bias = None
        self.last_hotspot_distribution = None
        self.last_coverage_loss = None

    def _directed_edges(self, edge_index):
        if edge_index.numel() == 0:
            return edge_index
        src, dst = edge_index
        if self.direction == "bottom_up":
            return torch.stack((dst, src), dim=0)
        if self.direction == "undirected":
            return torch.cat((edge_index, torch.stack((dst, src), dim=0)), dim=1)
        return edge_index

    def _neighbor_mean(self, values, edge_index):
        """Row-normalized aggregation with explicit self loops."""
        num_nodes = values.size(0)
        if num_nodes == 0:
            return values
        edges = self._directed_edges(edge_index)
        loop = torch.arange(num_nodes, device=values.device, dtype=torch.long)
        if edges.numel() == 0:
            src = loop
            dst = loop
        else:
            src = torch.cat((edges[0].long(), loop), dim=0)
            dst = torch.cat((edges[1].long(), loop), dim=0)
        aggregated = values.new_zeros(num_nodes)
        aggregated.index_add_(0, dst, values[src])
        degree = values.new_zeros(num_nodes)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=values.dtype))
        return aggregated / degree.clamp_min(1.0)

    def _normalize_per_graph(self, field, batch):
        if field.numel() == 0:
            return field
        field_column = field.unsqueeze(-1)
        graph_min = global_min_pool(field_column, batch).squeeze(-1)
        graph_max = global_max_pool(field_column, batch).squeeze(-1)
        span = graph_max - graph_min
        normalized = (field - graph_min[batch]) / span[batch].clamp_min(self.eps)
        # A one-node graph (or a perfectly flat field) is still valid evidence;
        # use a neutral 0.5 rather than erasing it to zero.
        flat = span[batch] <= self.eps
        return torch.where(flat, normalized.new_full(normalized.shape, 0.5), normalized)

    def forward(self, change_nodes, edge_index, batch):
        if change_nodes is None:
            raise RuntimeError("ConflictHotspotField requires Change node features")
        local = F.softplus(self.intensity(change_nodes).squeeze(-1))
        field = local
        for _ in range(self.steps):
            field = (1.0 - self.alpha) * local + self.alpha * self._neighbor_mean(
                field, edge_index
            )
        normalized = self._normalize_per_graph(field, batch.long())
        pool_multiplier = 1.0 + self.change_pool_scale * normalized
        attention_bias = self.semantic_tree_bias_scale * normalized

        self.last_local_intensity = local
        self.last_field_intensity = field
        self.last_normalized_field = normalized
        self.last_pool_multiplier = pool_multiplier
        self.last_attention_bias = attention_bias
        self.last_hotspot_distribution = None
        self.last_coverage_loss = None
        return {
            "local_intensity": local,
            "field_intensity": field,
            "normalized_field": normalized,
            "pool_multiplier": pool_multiplier,
            "attention_bias": attention_bias,
        }

    def coverage_loss(self, normalized_field, batch, attention_probability, valid_mask):
        """One-way hotspot-to-readout coordination loss.

        Hotspots act as a stopped teacher by default.  Query attention remains
        free to cover additional globally useful nodes.
        """
        if self.lambda_coverage <= 0.0:
            loss = normalized_field.new_zeros(())
            self.last_coverage_loss = loss
            return loss
        if attention_probability is None or valid_mask is None:
            raise RuntimeError(
                "Conflict-hotspot coverage requires Semantic-tree attention"
            )
        dense_field, field_mask = to_dense_batch(normalized_field, batch.long())
        mask = valid_mask & field_mask
        teacher_logits = dense_field / self.coverage_temperature
        teacher_logits = teacher_logits.masked_fill(~mask, -1e9)
        teacher = F.softmax(teacher_logits, dim=-1)
        if self.detach_coverage_teacher:
            teacher = teacher.detach()
        student = attention_probability.mean(dim=1).clamp_min(self.eps)
        per_graph = -(teacher * student.log()).sum(dim=-1)
        loss = self.lambda_coverage * per_graph.mean()
        self.last_hotspot_distribution = teacher
        self.last_coverage_loss = loss
        return loss
