"""Experimental EBGCN-BiGCN with same/different dual-subgraph routing."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from model.EBGCN import (
    _EdgeInference,
    _apply_batch_norm,
    _bottom_up_edges,
    _root_features,
    _top_down_edges,
)
from model.state_aux_samediff import StateAuxSameDiffEnhancer


class _DirectionalBiGCNDualSubgraphEncoder(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_features,
        output_features,
        edge_num,
        infer_edges,
        args,
    ):
        super().__init__()
        self.infer_edges = infer_edges
        self.conv1 = GCNConv(in_feats, hidden_features)
        self.edge_infer = _EdgeInference(hidden_features, edge_num)
        self.enhancer = StateAuxSameDiffEnhancer(hidden_features, args)
        self.bn1 = nn.BatchNorm1d(in_feats + hidden_features)
        self.conv2 = GCNConv(in_feats + hidden_features, output_features)

    def auxiliary_loss(self):
        loss = self.enhancer.auxiliary_loss()
        if loss is None:
            return self.conv1.lin.weight.new_zeros(())
        return loss

    def _finish_view(self, h, x_input, edge_index, edge_weight, data):
        x = torch.cat((h, _root_features(x_input, data)), dim=1)
        x = F.relu(_apply_batch_norm(self.bn1, x))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = torch.cat((x, _root_features(h, data)), dim=1)
        return global_mean_pool(x, data.batch)

    def forward(self, data, edge_index):
        x_input = data.x.float()
        h = F.relu(self.conv1(x_input, edge_index))
        if self.infer_edges:
            edge_loss, edge_weight = self.edge_infer(h, edge_index)
        else:
            edge_loss, edge_weight = None, None

        same_h, diff_h = self.enhancer(
            h,
            edge_index,
            node_state=getattr(data, 'node_state', None),
            edge_stance=getattr(data, 'edge_stance', None),
            return_views=True,
            tree_edge_index=_top_down_edges(data),
            batch=data.batch,
        )
        same_graph = self._finish_view(
            same_h, x_input, edge_index, edge_weight, data
        )
        diff_graph = self._finish_view(
            diff_h, x_input, edge_index, edge_weight, data
        )
        return (same_graph, diff_graph), edge_loss


class EBGCNBiGCNStateAuxSameDiff(nn.Module):
    """Separate BiGCN-backbone EBGCN dual-subgraph experiment."""

    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        hidden_features = int(getattr(args, 'ebgcn_hidden_dim', hidden_dim))
        output_features = int(getattr(args, 'ebgcn_output_dim', hidden_features))
        edge_num = int(getattr(args, 'ebgcn_edge_num', 2))
        if edge_num < 2:
            raise ValueError('ebgcn_edge_num must be at least 2.')

        self.TDrumorGCN = _DirectionalBiGCNDualSubgraphEncoder(
            in_feats,
            hidden_features,
            output_features,
            edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_td', True)),
            args,
        )
        self.BUrumorGCN = _DirectionalBiGCNDualSubgraphEncoder(
            in_feats,
            hidden_features,
            output_features,
            edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_bu', True)),
            args,
        )
        graph_dim = (hidden_features + output_features) * 2
        self.view_fusion = nn.Linear(graph_dim * 2, graph_dim)
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
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = F.log_softmax(self.fc(x), dim=1)
        return out, td_edge_loss, bu_edge_loss
