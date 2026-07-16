"""LIRS-inspired spuriosity-aware EBGCN.

This model adapts three ideas from LIRS to propagation-tree rumor detection:

1. a deliberately shortcut-biased encoder learns the easy/spurious component;
2. edge uncertainty is inferred from the residual (invariant) representation;
3. graph representations are regularized with biased infomax, HSIC, and
   class-conditional spurious prototypes.

The implementation is end-to-end and does not require LIRS' offline GSAT
explanations or precomputed KMeans assignments.  ``lirs_ebgcn_backbone`` can
be either ``bigcn`` or ``resgcn``.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from model.EBGCN import (
    _EdgeInference,
    _apply_batch_norm,
    _bottom_up_edges,
    _root_features,
    _top_down_edges,
)
from model.EIN_ResGCN import GCNConv as ResGCNConv


def _segment_mean(x, batch):
    return global_mean_pool(x, batch)


def _class_center(x, labels):
    """Center samples within their class without assuming balanced batches."""
    centered = torch.zeros_like(x)
    for label in labels.unique():
        mask = labels == label
        if int(mask.sum()) > 1:
            centered[mask] = x[mask] - x[mask].mean(dim=0, keepdim=True)
    return centered


def _linear_hsic(x, y):
    """Stable linear-kernel HSIC used for mini-batch representation removal."""
    if x.size(0) <= 1:
        return x.sum() * 0.0
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    covariance = x.t().matmul(y) / max(1, x.size(0) - 1)
    return covariance.square().mean()


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


class _SpuriositySeparator(nn.Module):
    """Split a mixed node representation into spurious and residual views."""

    def __init__(self, hidden_dim, args):
        super().__init__()
        dropout = float(getattr(args, 'lirs_spurious_dropout', 0.1))
        self.removal_strength = float(
            getattr(args, 'lirs_removal_strength', 0.5)
        )
        self.detach_projection = bool(
            getattr(args, 'lirs_detach_spurious_projection', True)
        )
        self.spurious_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.invariant_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.invariant_norm = nn.LayerNorm(hidden_dim)

    def forward(self, mixed):
        spurious = self.spurious_encoder(mixed)
        projection_basis = spurious.detach() if self.detach_projection else spurious
        basis = F.normalize(projection_basis, p=2, dim=-1, eps=1e-8)
        coefficient = (mixed * basis).sum(dim=-1, keepdim=True)
        gate = self.gate(torch.cat((mixed, spurious), dim=-1))
        residual = mixed - self.removal_strength * gate * coefficient * basis
        invariant = self.invariant_norm(residual + self.invariant_refine(residual))
        return invariant, spurious, gate


class _ClassConditionalPrototypeBank(nn.Module):
    """Online class-wise spurious clusters used by an adversarial head.

    Official LIRS obtains class-conditional cluster labels offline.  Here the
    same principle is made compatible with the project's end-to-end trainer by
    maintaining an EMA prototype bank for each rumor class.
    """

    def __init__(self, num_classes, num_clusters, hidden_dim, momentum):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_clusters = int(num_clusters)
        self.momentum = float(momentum)
        self.register_buffer(
            'prototypes',
            torch.zeros(self.num_classes, self.num_clusters, hidden_dim),
        )
        self.register_buffer(
            'prototype_counts',
            torch.zeros(self.num_classes, self.num_clusters),
        )

    @torch.no_grad()
    def assign_and_update(self, embeddings, labels, update=True):
        embeddings = F.normalize(embeddings.detach(), p=2, dim=-1, eps=1e-8)
        assignments = torch.zeros_like(labels, dtype=torch.long)
        for sample_index in range(embeddings.size(0)):
            label = int(labels[sample_index].item())
            if label < 0 or label >= self.num_classes:
                continue
            empty = (self.prototype_counts[label] == 0).nonzero(
                as_tuple=False
            ).view(-1)
            if empty.numel() > 0:
                cluster = int(empty[0].item())
            else:
                similarity = self.prototypes[label].matmul(
                    embeddings[sample_index]
                )
                cluster = int(similarity.argmax().item())
            assignments[sample_index] = cluster
            if update:
                if self.prototype_counts[label, cluster] == 0:
                    new_value = embeddings[sample_index]
                else:
                    new_value = (
                        self.momentum * self.prototypes[label, cluster]
                        + (1.0 - self.momentum) * embeddings[sample_index]
                    )
                    new_value = F.normalize(new_value, p=2, dim=0, eps=1e-8)
                self.prototypes[label, cluster].copy_(new_value)
                self.prototype_counts[label, cluster].add_(1)
        return assignments


class _LIRSDirectionalBase(nn.Module):
    def __init__(self, hidden_dim, edge_num, infer_edges, args):
        super().__init__()
        self.infer_edges = infer_edges
        self.separator = _SpuriositySeparator(hidden_dim, args)
        self.edge_infer = _EdgeInference(hidden_dim, edge_num)
        self.edge_suppression = float(
            getattr(args, 'lirs_edge_spurious_suppression', 0.5)
        )
        self.target_ratio = float(getattr(args, 'lirs_spurious_ratio', 0.3))

    def _infer_edge(self, invariant, gate, edge_index):
        if not self.infer_edges:
            return None, None
        edge_loss, edge_weight = self.edge_infer(invariant, edge_index)
        if edge_index.numel() > 0 and self.edge_suppression > 0:
            row, col = edge_index
            contamination = 0.5 * (gate[row, 0] + gate[col, 0])
            attenuation = (1.0 - self.edge_suppression * contamination).clamp(
                min=0.05, max=1.0
            )
            edge_weight = edge_weight * attenuation
        return edge_loss, edge_weight

    def _biased_infomax(self, invariant, spurious, gate, batch):
        if spurious.size(0) == 0:
            return spurious.sum() * 0.0
        graph_summary = torch.sigmoid(_segment_mean(spurious, batch))[batch]
        scale = math.sqrt(max(1, spurious.size(-1)))
        positive = (spurious * graph_summary).sum(dim=-1) / scale
        invariant_score = (invariant * graph_summary).sum(dim=-1) / scale

        if self.training and spurious.size(0) > 1:
            permutation = torch.randperm(spurious.size(0), device=spurious.device)
        else:
            permutation = torch.arange(
                spurious.size(0) - 1, -1, -1, device=spurious.device
            )
        negative = (spurious[permutation] * graph_summary).sum(dim=-1) / scale
        weights = gate[:, 0].detach().clamp_min(1e-3)
        positive_loss = -(
            weights * F.logsigmoid(positive)
        ).sum() / weights.sum()
        negative_loss = -F.logsigmoid(-negative).mean()
        removal_loss = -F.logsigmoid(-invariant_score).mean()
        return positive_loss + negative_loss + removal_loss

    def _gate_balance(self, gate):
        return (gate.mean() - self.target_ratio).square()


class _LIRSBiGCNDirection(_LIRSDirectionalBase):
    def __init__(
        self,
        in_feats,
        hidden_dim,
        output_dim,
        edge_num,
        infer_edges,
        args,
    ):
        super().__init__(hidden_dim, edge_num, infer_edges, args)
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.bn1 = nn.BatchNorm1d(in_feats + hidden_dim)
        self.conv2 = GCNConv(in_feats + hidden_dim, output_dim)

    def forward(self, data, edge_index):
        x_input = data.x.float()
        mixed = F.relu(self.conv1(x_input, edge_index))
        invariant, spurious, gate = self.separator(mixed)
        edge_loss, edge_weight = self._infer_edge(invariant, gate, edge_index)

        x = torch.cat((invariant, _root_features(x_input, data)), dim=1)
        x = F.relu(_apply_batch_norm(self.bn1, x))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = torch.cat((x, _root_features(invariant, data)), dim=1)
        invariant_graph = global_mean_pool(x, data.batch)
        spurious_graph = global_mean_pool(spurious, data.batch)
        return invariant_graph, spurious_graph, {
            'edge_loss': edge_loss,
            'infomax_loss': self._biased_infomax(
                invariant, spurious, gate, data.batch
            ),
            'gate_balance_loss': self._gate_balance(gate),
            'mean_spurious_gate': gate.mean().detach(),
        }


class _LIRSResGCNDirection(_LIRSDirectionalBase):
    def __init__(self, in_feats, hidden_dim, edge_num, infer_edges, args):
        super().__init__(hidden_dim, edge_num, infer_edges, args)
        self.residual = bool(getattr(args, 'skip_connection', True))
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.bn_feat = nn.BatchNorm1d(in_feats)
        self.conv_feat = ResGCNConv(in_feats, hidden_dim, gfn=True)
        num_layers = int(
            getattr(
                args,
                'ebgcn_resgcn_num_conv_layers',
                getattr(args, 'n_layers_conv', 3),
            )
        )
        if num_layers < 1:
            raise ValueError('LIRS-EBGCN ResGCN requires at least one convolution.')
        edge_norm = bool(getattr(args, 'edge_norm', True))
        self.bns_conv = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.convs = nn.ModuleList(
            [
                ResGCNConv(hidden_dim, hidden_dim, edge_norm=edge_norm)
                for _ in range(num_layers)
            ]
        )
        global_pool = str(getattr(args, 'global_pool', 'sum'))
        self.pool = global_add_pool if 'sum' in global_pool else global_mean_pool
        self.gating = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            if 'gating' in global_pool
            else None
        )

    def forward(self, data, edge_index):
        mixed = F.relu(
            self.conv_feat(
                _apply_batch_norm(self.bn_feat, data.x.float()), edge_index
            )
        )
        invariant, spurious, gate = self.separator(mixed)
        edge_loss, edge_weight = self._infer_edge(invariant, gate, edge_index)
        x = invariant
        for batch_norm, conv in zip(self.bns_conv, self.convs):
            update = F.relu(
                conv(
                    _apply_batch_norm(batch_norm, x),
                    edge_index,
                    edge_weight=edge_weight,
                )
            )
            x = x + update if self.residual else update
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        pool_gate = 1 if self.gating is None else self.gating(x)
        invariant_graph = self.pool(x * pool_gate, data.batch)
        spurious_graph = self.pool(spurious, data.batch)
        return invariant_graph, spurious_graph, {
            'edge_loss': edge_loss,
            'infomax_loss': self._biased_infomax(
                invariant, spurious, gate, data.batch
            ),
            'gate_balance_loss': self._gate_balance(gate),
            'mean_spurious_gate': gate.mean().detach(),
        }


class LIRSEBGCN(nn.Module):
    """Spuriosity-removed edge-uncertainty model with selectable backbone."""

    SUPPORTED_BACKBONES = {'bigcn', 'resgcn'}

    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.num_classes = int(num_classes)
        self.backbone = str(
            getattr(args, 'lirs_ebgcn_backbone', 'bigcn')
        ).strip().lower()
        if self.backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                'lirs_ebgcn_backbone must be one of {}, got {!r}.'.format(
                    sorted(self.SUPPORTED_BACKBONES), self.backbone
                )
            )

        hidden = int(getattr(args, 'ebgcn_hidden_dim', hidden_dim))
        output = int(getattr(args, 'ebgcn_output_dim', hidden))
        edge_num = int(getattr(args, 'ebgcn_edge_num', 2))
        if edge_num < 2:
            raise ValueError('ebgcn_edge_num must be at least 2.')
        infer_td = bool(getattr(args, 'ebgcn_edge_infer_td', True))
        infer_bu = bool(getattr(args, 'ebgcn_edge_infer_bu', True))

        if self.backbone == 'bigcn':
            direction_factory = lambda infer: _LIRSBiGCNDirection(
                in_feats, hidden, output, edge_num, infer, args
            )
            graph_dim = 2 * (hidden + output)
        else:
            direction_factory = lambda infer: _LIRSResGCNDirection(
                in_feats, hidden, edge_num, infer, args
            )
            graph_dim = 2 * hidden
        self.TDrumorGCN = direction_factory(infer_td)
        self.BUrumorGCN = direction_factory(infer_bu)
        self.graph_dim = graph_dim
        spurious_dim = 2 * hidden
        self.spurious_projector = (
            nn.Identity()
            if spurious_dim == graph_dim
            else nn.Linear(spurious_dim, graph_dim)
        )
        self.classifier = nn.Linear(graph_dim, num_classes)
        self.spurious_classifier = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.ReLU(),
            nn.Dropout(float(getattr(args, 'dropout', 0.0))),
            nn.Linear(hidden, num_classes),
        )

        num_clusters = int(getattr(args, 'lirs_num_clusters', 2))
        if num_clusters < 1:
            raise ValueError('lirs_num_clusters must be at least 1.')
        self.num_clusters = num_clusters
        self.prototype_bank = _ClassConditionalPrototypeBank(
            num_classes,
            num_clusters,
            graph_dim,
            float(getattr(args, 'lirs_prototype_momentum', 0.9)),
        )
        self.cluster_classifier = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_clusters),
        )
        self.grl_scale = float(getattr(args, 'lirs_cluster_grl_scale', 1.0))

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    @staticmethod
    def _mean_directional(aux_td, aux_bu, key, reference):
        values = [aux.get(key) for aux in (aux_td, aux_bu)]
        values = [value for value in values if value is not None]
        return torch.stack(values).mean() if values else reference.sum() * 0.0

    def forward(self, data):
        td_graph, td_spurious, td_aux = self.TDrumorGCN(
            data, _top_down_edges(data)
        )
        bu_graph, bu_spurious, bu_aux = self.BUrumorGCN(
            data, _bottom_up_edges(data)
        )
        invariant_graph = torch.cat((bu_graph, td_graph), dim=-1)
        spurious_graph = self.spurious_projector(
            torch.cat((bu_spurious, td_spurious), dim=-1)
        )
        logits = self.classifier(invariant_graph)
        out = F.log_softmax(logits, dim=-1)

        labels = data.y.view(-1).long()
        spurious_logits = self.spurious_classifier(spurious_graph)
        shortcut_loss = F.cross_entropy(spurious_logits, labels)
        normalized_invariant = F.normalize(
            invariant_graph, p=2, dim=-1, eps=1e-8
        )
        normalized_spurious = F.normalize(
            spurious_graph.detach(), p=2, dim=-1, eps=1e-8
        )
        independence_loss = _linear_hsic(
            normalized_invariant, normalized_spurious
        )
        conditional_independence_loss = _linear_hsic(
            _class_center(normalized_invariant, labels),
            _class_center(normalized_spurious, labels),
        )

        if self.num_clusters > 1:
            cluster_targets = self.prototype_bank.assign_and_update(
                spurious_graph,
                labels,
                update=self.training,
            )
            reversed_graph = _GradientReversal.apply(
                invariant_graph, self.grl_scale
            )
            cluster_loss = F.cross_entropy(
                self.cluster_classifier(reversed_graph), cluster_targets
            )
        else:
            cluster_loss = invariant_graph.sum() * 0.0

        edge_loss = self._mean_directional(
            td_aux, bu_aux, 'edge_loss', out
        )
        auxiliary = {
            'edge_loss': edge_loss,
            'shortcut_loss': shortcut_loss,
            'independence_loss': independence_loss,
            'conditional_independence_loss': conditional_independence_loss,
            'cluster_loss': cluster_loss,
            'infomax_loss': self._mean_directional(
                td_aux, bu_aux, 'infomax_loss', out
            ),
            'gate_balance_loss': self._mean_directional(
                td_aux, bu_aux, 'gate_balance_loss', out
            ),
            'mean_spurious_gate': 0.5
            * (
                td_aux['mean_spurious_gate']
                + bu_aux['mean_spurious_gate']
            ),
            'invariant_graph': invariant_graph,
            'spurious_graph': spurious_graph,
        }
        return out, auxiliary
