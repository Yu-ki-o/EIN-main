"""SHPA from Sections 2.2--2.4 of ``my model.pdf``.

Gemma 2-9B labels edges offline: support=0, deny=1. Labels are required at
training AND inference. The method equations define two independent GCNs,
attention pools and channel alignment, without an original-graph branch.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax, to_undirected

from model.DualBackboneOnly import (
    BiGCN_BackboneOnly,
    ResGCN_BackboneOnly,
)


def channel_contrastive_loss(support, deny, temperature=0.2):
    """Equation (3), with same-channel positives and cross-channel negatives.

    Self-pairs are excluded; the opposite channel of the same event remains
    a negative. No rumor labels are used. B=1 has no positives and returns 0.
    """
    if support.ndim != 2 or support.shape != deny.shape:
        raise ValueError("support and deny must have matching [B, hidden_dim] shapes")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    batch_size = support.size(0)
    if batch_size < 2:
        return (support.sum() + deny.sum()) * 0.0
    z = F.normalize(torch.cat((support, deny), dim=0), p=2, dim=-1)
    scores = z @ z.t() / temperature
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    channels = torch.arange(2, device=z.device).repeat_interleave(batch_size)
    positives = (channels[:, None] == channels[None, :]) & ~diagonal
    log_denominator = scores.masked_fill(diagonal, -torch.inf).logsumexp(dim=1)
    positive_mean = scores.masked_fill(~positives, 0).sum(dim=1) / (batch_size - 1)
    return (log_denominator - positive_mean).mean()


class StanceGCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_feats if i == 0 else hidden_dim, hidden_dim,
                    cached=False, add_self_loops=True, normalize=True)
            for i in range(num_layers)
        ])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class _EncoderOnlyMixin:
    """Reuse an existing project backbone without allocating its classifier."""

    def _build_classifier(self, args):
        self.fusion = nn.Identity()
        self.classifier = nn.Identity()

    def forward(self, data):
        return self._encode_nodes(data)


class StanceBiGCN(_EncoderOnlyMixin, BiGCN_BackboneOnly):
    """Existing BiGCN node encoder adapted for one SHPA relation graph."""


class StanceResGCN(_EncoderOnlyMixin, ResGCN_BackboneOnly):
    """Existing ResGCN node encoder adapted for one SHPA relation graph."""

    def _build_backbone(self, in_feats, hid_feats, out_feats, args):
        super()._build_backbone(in_feats, hid_feats, out_feats, args)
        # The matched ResGCN feature projection runs in GFN mode, whose
        # implementation returns before adding bias. Do not expose that
        # unreachable parameter to the SHPA optimizer.
        self.conv_feat.register_parameter("bias", None)


def build_stance_encoder(backbone, in_feats, hidden_dim, num_layers,
                         dropout, args, device):
    if backbone == "gcn":
        return StanceGCN(in_feats, hidden_dim, num_layers, dropout)

    backbone_args = copy.copy(args) if args is not None else SimpleNamespace()
    # Keep one layer-count field across all three choices. BiGCN requires at
    # least two layers internally, matching its existing project implementation.
    backbone_args.n_layers_conv = num_layers
    backbone_args.dropout = dropout
    backbone_class = {
        "resgcn": StanceResGCN,
        "bigcn": StanceBiGCN,
    }[backbone]
    return backbone_class(
        in_feats,
        hidden_dim,
        hidden_dim,
        1,
        backbone_args,
        device,
    )


class AttentionPool(nn.Module):
    """Learnable scalar attention normalized separately within each graph."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        weights = softmax(self.gate(x), batch)
        return global_add_pool(weights * x, batch)


class SHPA(nn.Module):
    """Return (log_probs, U, S, D) for the existing EINTrainer interface.

    U/S/D are zero placeholders, not epidemiological states. The auxiliary
    loss hook returns lambda * Lctr during training only.
    """
    def __init__(self, in_feats, hidden_dim=128, num_classes=2, args=None,
                 device=None):
        super().__init__()
        num_layers = int(getattr(args, "shpa_num_layers", 3))
        dropout = float(getattr(args, "dropout", 0.0))
        self.backbone = str(getattr(args, "shpa_backbone", "gcn")).strip().lower()
        self.temperature = float(getattr(args, "shpa_temperature", 0.2))
        dataset = str(getattr(args, "dataset", "DRWeibo")).lower()
        default_lambda = {"pheme": 2.0, "weibo": 0.5, "drweibo": 1.0}.get(dataset, 1.0)
        self.lambda_contrastive = float(getattr(args, "shpa_lambda_contrastive", default_lambda))
        self.max_hop = int(getattr(args, "max_hop", 1))
        if min(in_feats, hidden_dim, num_classes, num_layers, self.max_hop) < 1:
            raise ValueError("dimensions, classes, layers and max_hop must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.backbone not in {"gcn", "resgcn", "bigcn"}:
            raise ValueError("shpa_backbone must be one of: gcn, resgcn, bigcn")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("shpa_temperature must be finite and positive")
        if not math.isfinite(self.lambda_contrastive) or self.lambda_contrastive < 0:
            raise ValueError("shpa_lambda_contrastive must be finite and nonnegative")
        self.support_encoder = build_stance_encoder(
            self.backbone, in_feats, hidden_dim, num_layers, dropout,
            args, device,
        )
        self.deny_encoder = build_stance_encoder(
            self.backbone, in_feats, hidden_dim, num_layers, dropout,
            args, device,
        )
        self.support_pool = AttentionPool(hidden_dim)
        self.deny_pool = AttentionPool(hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self._contrastive_loss = None

    @staticmethod
    def stance_graphs(data):
        # Never mix an edge tensor with labels from a differently ordered one.
        edge_index = getattr(data, "directed_edge_index", None)
        stance = getattr(data, "directed_edge_stance", None)
        if edge_index is None or stance is None:
            edge_index = data.edge_index
            stance = getattr(data, "edge_stance", None)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long")
        if stance is None:
            if edge_index.size(1):
                raise ValueError("SHPA requires LLM edge_stance labels (0=support, 1=deny)")
            stance = edge_index.new_empty(0)
        stance = stance.reshape(-1).to(edge_index.device)
        if stance.numel() != edge_index.size(1):
            raise ValueError("Each edge must have one aligned stance label")
        if not bool(((stance == 0) | (stance == 1)).all()):
            raise ValueError("SHPA requires complete 0/1 edge stances; annotate missing labels with Gemma first")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= data.x.size(0)):
            raise ValueError("edge_index contains an invalid node index")
        batch = getattr(data, "batch", None)
        if batch is not None and bool((batch[edge_index[0]] != batch[edge_index[1]]).any()):
            raise ValueError("Edges must not connect different graphs in a batch")
        # Equation (2) defines undirected sets; coalesce repeated edges.
        return tuple(to_undirected(edge_index[:, stance == label], num_nodes=data.x.size(0))
                     for label in (0, 1))

    def encode(self, data):
        if data.x.ndim != 2 or data.x.size(0) == 0:
            raise ValueError("SHPA requires at least one node with a feature vector")
        support_edges, deny_edges = self.stance_graphs(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        if self.backbone == "gcn":
            support = self.support_encoder(data.x, support_edges)
            deny = self.deny_encoder(data.x, deny_edges)
        else:
            support_data = copy.copy(data)
            support_data.edge_index = support_edges
            support_data.batch = batch
            deny_data = copy.copy(data)
            deny_data.edge_index = deny_edges
            deny_data.batch = batch
            support = self.support_encoder(support_data)
            deny = self.deny_encoder(deny_data)
        return self.support_pool(support, batch), self.deny_pool(deny, batch)

    def forward(self, data):
        self._contrastive_loss = None
        support, deny = self.encode(data)
        output = F.log_softmax(self.classifier(torch.cat((support, deny), dim=-1)), dim=-1)
        if self.training and self.lambda_contrastive > 0:
            self._contrastive_loss = channel_contrastive_loss(support, deny, self.temperature)
        dummy = output.new_zeros((output.size(0), self.max_hop, 1))
        return output, dummy, dummy, dummy

    def auxiliary_loss(self):
        if not self.training or self._contrastive_loss is None:
            return self.classifier.weight.new_zeros(())
        return self.lambda_contrastive * self._contrastive_loss

    def classification_loss(self, output, target):
        return F.nll_loss(output, target.reshape(-1).long())

    def physics_loss(self, U, S, D, true_state):
        return self.classifier.weight.new_zeros(())

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr,
                                weight_decay=getattr(args, "weight_decay", 0.0))
