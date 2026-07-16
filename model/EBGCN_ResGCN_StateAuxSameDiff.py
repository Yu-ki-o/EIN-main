"""Experimental EBGCN-ResGCN with same/different dual-subgraph routing."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from model.EBGCN import (
    _EdgeInference,
    _apply_batch_norm,
    _bottom_up_edges,
    _top_down_edges,
)
from model.EIN_ResGCN import GCNConv as ResGCNConv
from model.state_aux_samediff import StateAuxSameDiffEnhancer


class _DirectionalDualSubgraphEncoder(nn.Module):
    def __init__(self, in_feats, hidden_features, edge_num, infer_edges, args):
        super().__init__()
        self.infer_edges = infer_edges
        self.residual = bool(getattr(args, 'skip_connection', True))
        self.bn_feat = nn.BatchNorm1d(in_feats)
        self.conv_feat = ResGCNConv(in_feats, hidden_features, gfn=True)
        self.edge_infer = _EdgeInference(hidden_features, edge_num)

        num_conv_layers = int(
            getattr(
                args,
                'ebgcn_resgcn_num_conv_layers',
                getattr(args, 'n_layers_conv', 3),
            )
        )
        if num_conv_layers < 1:
            raise ValueError('At least one residual graph convolution is required.')
        edge_norm = bool(getattr(args, 'edge_norm', True))
        self.bns_conv = nn.ModuleList(
            [nn.BatchNorm1d(hidden_features) for _ in range(num_conv_layers)]
        )
        self.convs = nn.ModuleList(
            [
                ResGCNConv(hidden_features, hidden_features, edge_norm=edge_norm)
                for _ in range(num_conv_layers)
            ]
        )
        self.enhancers = nn.ModuleList(
            [
                StateAuxSameDiffEnhancer(hidden_features, args)
                for _ in range(num_conv_layers)
            ]
        )

        global_pool = str(getattr(args, 'global_pool', 'sum'))
        if 'sum' in global_pool:
            self.global_pool = global_add_pool
        elif 'mean' in global_pool:
            self.global_pool = global_mean_pool
        else:
            raise ValueError('global_pool must contain "sum" or "mean".')
        self.gating = (
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1),
                nn.Sigmoid(),
            )
            if 'gating' in global_pool
            else None
        )

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)

    def auxiliary_loss(self):
        losses = [
            loss
            for enhancer in self.enhancers
            for loss in [enhancer.auxiliary_loss()]
            if loss is not None
        ]
        if not losses:
            return self.conv_feat.weight.new_zeros(())
        return torch.stack(losses).mean()

    def _pool(self, x, batch):
        gate = 1 if self.gating is None else self.gating(x)
        return self.global_pool(x * gate, batch)

    def forward(self, data, edge_index):
        x = F.relu(
            self.conv_feat(
                _apply_batch_norm(self.bn_feat, data.x.float()), edge_index
            )
        )
        if self.infer_edges:
            edge_loss, edge_weight = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_weight = None, None

        node_state = getattr(data, 'node_state', None)
        edge_stance = getattr(data, 'edge_stance', None)
        tree_edge_index = _top_down_edges(data)
        x_diff = x
        for batch_norm, conv, enhancer in zip(
            self.bns_conv, self.convs, self.enhancers
        ):
            conv_out = F.relu(
                conv(
                    _apply_batch_norm(batch_norm, x),
                    edge_index,
                    edge_weight=edge_weight,
                )
            )
            same_out, diff_out = enhancer(
                conv_out,
                edge_index,
                node_state=node_state,
                edge_stance=edge_stance,
                return_views=True,
                tree_edge_index=tree_edge_index,
                batch=data.batch,
            )
            if self.residual:
                x_diff = x + diff_out
                x = x + same_out
            else:
                x_diff = diff_out
                x = same_out

        same_graph = self._pool(x, data.batch)
        diff_graph = self._pool(x_diff, data.batch)
        return (same_graph, diff_graph), edge_loss


class EBGCNResGCNStateAuxSameDiff(nn.Module):
    """Separate experimental model; the base EBGCN implementations are untouched."""

    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_features = int(getattr(args, 'ebgcn_hidden_dim', hidden_dim))
        self.edge_num = int(getattr(args, 'ebgcn_edge_num', 2))
        if self.edge_num < 2:
            raise ValueError('ebgcn_edge_num must be at least 2.')

        self.TDrumorGCN = _DirectionalDualSubgraphEncoder(
            in_feats,
            self.hidden_features,
            self.edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_td', True)),
            args,
        )
        self.BUrumorGCN = _DirectionalDualSubgraphEncoder(
            in_feats,
            self.hidden_features,
            self.edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_bu', True)),
            args,
        )

        graph_dim = self.hidden_features * 2
        self.view_fusion = nn.Linear(graph_dim * 2, graph_dim)
        num_fc_layers = int(getattr(args, 'n_layers_fc', 2))
        self.bns_fc = nn.ModuleList(
            [nn.BatchNorm1d(graph_dim) for _ in range(max(0, num_fc_layers - 1))]
        )
        self.lins = nn.ModuleList(
            [nn.Linear(graph_dim, graph_dim) for _ in range(max(0, num_fc_layers - 1))]
        )
        self.bn_hidden = nn.BatchNorm1d(graph_dim)
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.fc = nn.Linear(graph_dim, num_classes)

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def auxiliary_loss(self):
        losses = [
            branch.auxiliary_loss()
            for branch in (self.TDrumorGCN, self.BUrumorGCN)
        ]
        return torch.stack(losses).mean()

    def forward(self, data):
        (td_same, td_diff), td_edge_loss = self.TDrumorGCN(
            data, _top_down_edges(data)
        )
        (bu_same, bu_diff), bu_edge_loss = self.BUrumorGCN(
            data, _bottom_up_edges(data)
        )
        same_x = torch.cat((bu_same, td_same), dim=1)
        diff_x = torch.cat((bu_diff, td_diff), dim=1)
        x = F.relu(self.view_fusion(torch.cat((same_x, diff_x), dim=1)))
        for batch_norm, linear in zip(self.bns_fc, self.lins):
            x = F.relu(linear(_apply_batch_norm(batch_norm, x)))
        x = _apply_batch_norm(self.bn_hidden, x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = F.log_softmax(self.fc(x), dim=1)
        return out, td_edge_loss, bu_edge_loss
