import sys
import os

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

from model.state_aux_samediff import StateAuxSameDiffEnhancer


def _root_extend(batch, values, device):
    out = torch.zeros(batch.size(0), values.size(1), device=device, dtype=values.dtype)
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    for num_batch in range(batch_size):
        index = torch.eq(batch, num_batch)
        out[index] = values[index][0]
    return out


class TDrumorGCN_StateAuxSameDiff(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, args, device):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.enhancer = StateAuxSameDiffEnhancer(hid_feats, args)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device
        self.dropout = float(getattr(args, 'dropout', 0.5))
        self.dual_view = bool(getattr(args, 'state_aux_dual_view', True))

    def _finish_branch(self, h, root_values, edge_index, batch):
        h1 = h
        root_extend = _root_extend(batch, root_values, self.device)
        h = torch.cat((h, root_extend), dim=1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        root_extend = _root_extend(batch, h1, self.device)
        h = torch.cat((h, root_extend), dim=1)
        return scatter_mean(h, batch, dim=0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        node_state = getattr(data, 'node_state', None)
        edge_stance = getattr(data, 'edge_stance', None)

        x1 = x.float()
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        if not self.dual_view:
            h = self.enhancer(
                h,
                edge_index,
                node_state=node_state,
                edge_stance=edge_stance,
                tree_edge_index=data.edge_index,
                batch=data.batch,
            )
            return self._finish_branch(h, x1, edge_index, data.batch)

        h_same, h_diff = self.enhancer(
            h,
            edge_index,
            node_state=node_state,
            edge_stance=edge_stance,
            return_views=True,
            tree_edge_index=data.edge_index,
            batch=data.batch,
        )
        return (
            self._finish_branch(h_same, x1, edge_index, data.batch),
            self._finish_branch(h_diff, x1, edge_index, data.batch),
        )

    def auxiliary_loss(self):
        loss = self.enhancer.auxiliary_loss()
        return loss


class BUrumorGCN_StateAuxSameDiff(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, args, device):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.enhancer = StateAuxSameDiffEnhancer(hid_feats, args)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device
        self.dropout = float(getattr(args, 'dropout', 0.5))
        self.dual_view = bool(getattr(args, 'state_aux_dual_view', True))

    def _finish_branch(self, h, root_values, edge_index, batch):
        h1 = h
        root_extend = _root_extend(batch, root_values, self.device)
        h = torch.cat((h, root_extend), dim=1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        root_extend = _root_extend(batch, h1, self.device)
        h = torch.cat((h, root_extend), dim=1)
        return scatter_mean(h, batch, dim=0)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]
        node_state = getattr(data, 'node_state', None)
        edge_stance = getattr(data, 'edge_stance', None)

        x1 = x.float()
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        if not self.dual_view:
            h = self.enhancer(
                h,
                edge_index,
                node_state=node_state,
                edge_stance=edge_stance,
                tree_edge_index=data.edge_index,
                batch=data.batch,
            )
            return self._finish_branch(h, x1, edge_index, data.batch)

        h_same, h_diff = self.enhancer(
            h,
            edge_index,
            node_state=node_state,
            edge_stance=edge_stance,
            return_views=True,
            tree_edge_index=data.edge_index,
            batch=data.batch,
        )
        return (
            self._finish_branch(h_same, x1, edge_index, data.batch),
            self._finish_branch(h_diff, x1, edge_index, data.batch),
        )

    def auxiliary_loss(self):
        loss = self.enhancer.auxiliary_loss()
        return loss


class BiGCN_StateAuxSameDiff(torch.nn.Module):
    """
    BiGCN baseline plus state-supervised same/different node enhancement.

    This variant intentionally removes the original EIN epidemiology branch:
    no U/S/D encoder, no xg fusion, and no physics loss.
    """

    def __init__(self, in_feats, hid_feats, out_feats, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.TDrumorGCN = TDrumorGCN_StateAuxSameDiff(in_feats, hid_feats, out_feats, args, device)
        self.BUrumorGCN = BUrumorGCN_StateAuxSameDiff(in_feats, hid_feats, out_feats, args, device)
        self.dropout = float(getattr(args, 'dropout', 0.5))
        self.dual_view = bool(getattr(args, 'state_aux_dual_view', True))
        graph_dim = (out_feats + hid_feats) * 2
        self.view_fusion = torch.nn.Linear(graph_dim * 2, graph_dim)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_classes)

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def physics_loss(self, U, S, D, true_state):
        return self.fc.weight.new_zeros(())

    def auxiliary_loss(self):
        losses = []
        for module in (self.TDrumorGCN, self.BUrumorGCN):
            loss = module.auxiliary_loss()
            if loss is not None:
                losses.append(loss)
        if not losses:
            return self.fc.weight.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, data):
        if self.dual_view:
            td_same, td_diff = self.TDrumorGCN(data)
            bu_same, bu_diff = self.BUrumorGCN(data)
            same_x = torch.cat((bu_same, td_same), dim=1)
            diff_x = torch.cat((bu_diff, td_diff), dim=1)
            x = torch.cat((same_x, diff_x), dim=1)
            x = F.relu(self.view_fusion(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            td_x = self.TDrumorGCN(data)
            bu_x = self.BUrumorGCN(data)
            x = torch.cat((bu_x, td_x), dim=1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x, None, None, None
