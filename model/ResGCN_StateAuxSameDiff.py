import sys
import os
from functools import partial

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import global_add_pool, global_mean_pool

from model.EIN_ResGCN import GCNConv
from model.state_aux_samediff import StateAuxSameDiffEnhancer


class ResGCN_StateAuxSameDiff(torch.nn.Module):
    """
    ResGCN baseline plus state-supervised same/different node enhancement.

    This variant removes the original EIN epidemiology branch and classifies
    directly from enhanced graph representations.
    """

    def __init__(self, dataset=None, num_classes=2, hidden=128, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True, args=None, device=None):
        super().__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.num_classes = num_classes
        self.conv_residual = residual
        self.fc_residual = False
        self.res_branch = res_branch
        self.collapse = collapse
        self.args = args
        self.device = device
        self.dropout = dropout
        self.dual_view = bool(getattr(args, 'state_aux_dual_view', True))

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        self.global_pool = global_add_pool if "sum" in global_pool else global_mean_pool

        gconv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = dataset.num_features

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)

        if "gating" in global_pool:
            self.gating = torch.nn.Sequential(
                Linear(hidden, hidden),
                torch.nn.ReLU(),
                Linear(hidden, 1),
                torch.nn.Sigmoid(),
            )
        else:
            self.gating = None

        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.enhancers = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(gconv(hidden, hidden))
            self.enhancers.append(StateAuxSameDiffEnhancer(hidden, args))

        self.bn_hidden = BatchNorm1d(hidden)
        self.view_fusion = Linear(hidden * 2, hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, self.num_classes)

        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0.0001)

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def physics_loss(self, U, S, D, true_state):
        return self.lin_class.weight.new_zeros(())

    def auxiliary_loss(self):
        losses = []
        for enhancer in self.enhancers:
            loss = enhancer.auxiliary_loss()
            if loss is not None:
                losses.append(loss)
        if not losses:
            return self.lin_class.weight.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_state = getattr(data, 'node_state', None)
        edge_stance = getattr(data, 'edge_stance', None)

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        x_diff = x
        for batch_norm, conv, enhancer in zip(self.bns_conv, self.convs, self.enhancers):
            x_ = batch_norm(x)
            x_ = F.relu(conv(x_, edge_index))
            if self.dual_view:
                x_same, x_diff_ = enhancer(
                    x_,
                    edge_index,
                    node_state=node_state,
                    edge_stance=edge_stance,
                    return_views=True,
                    batch=batch,
                )
            else:
                x_same = enhancer(
                    x_,
                    edge_index,
                    node_state=node_state,
                    edge_stance=edge_stance,
                    batch=batch,
                )
                x_diff_ = x_diff
            if self.conv_residual:
                x_diff = x + x_diff_ if self.dual_view else x_diff_
                x = x + x_same
            else:
                x_diff = x_diff_
                x = x_same

        gate = 1 if self.gating is None else self.gating(x)
        x_same = self.global_pool(x * gate, batch)
        if self.dual_view:
            diff_gate = 1 if self.gating is None else self.gating(x_diff)
            x_diff = self.global_pool(x_diff * diff_gate, batch)
            x = F.relu(self.view_fusion(torch.cat((x_same, x_diff), dim=-1)))
        else:
            x = x_same

        for batch_norm, lin in zip(self.bns_fc, self.lins):
            x_ = batch_norm(x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1), None, None, None

    def __repr__(self):
        return self.__class__.__name__
