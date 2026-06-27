import os
import sys
from functools import partial

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import global_add_pool, global_mean_pool

from model.EIN_ResGCN import GCNConv
from model.state_aux_samediff import StateAuxSameDiffEnhancer


class EINResGCNSameDiffFusion(torch.nn.Module):
    """
    EIN ResGCN with explicit same, diff, and epidemiology representations.

    The same/diff graph views are produced by the state-supervised routing
    module. The original EIN U/S/D recurrence produces the epidemiology view.
    Their graph-level representations are concatenated and projected before
    classification.
    """

    def __init__(
        self,
        dataset=None,
        num_classes=2,
        hidden=128,
        num_feat_layers=1,
        num_conv_layers=3,
        num_fc_layers=2,
        gfn=False,
        collapse=False,
        residual=False,
        res_branch="BNConvReLU",
        global_pool="sum",
        dropout=0,
        edge_norm=True,
        args=None,
        device=None,
    ):
        super().__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        assert args is not None, "args is required for EIN and same/diff settings"

        self.num_classes = num_classes
        self.hidden = hidden
        self.conv_residual = residual
        self.fc_residual = False
        self.res_branch = res_branch
        self.collapse = collapse
        self.args = args
        self.device = device
        self.dropout = dropout

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        self.global_pool = (
            global_add_pool if "sum" in global_pool else global_mean_pool
        )

        graph_conv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
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
            self.convs.append(graph_conv(hidden, hidden))
            self.enhancers.append(StateAuxSameDiffEnhancer(hidden, args))

        # Original EIN epidemiology encoder.
        self.W_u0 = Linear(1, hidden)
        self.W_s0 = Linear(1, hidden)
        self.W_d0 = Linear(1, hidden)
        self.W_u = Linear(hidden, hidden)
        self.W_s = Linear(hidden, hidden)
        self.W_d = Linear(hidden, hidden)
        self.W_x = Linear(hidden * 3, hidden)
        self.l_u = Linear(hidden, 1)
        self.l_s = Linear(hidden, 1)
        self.l_d = Linear(hidden, 1)

        if args.init_alpha == "random" and args.init_beta == "random":
            self.raw_alpha = torch.nn.Parameter(torch.rand(1))
            self.raw_beta = torch.nn.Parameter(torch.rand(1))
        else:
            self.raw_alpha = torch.nn.Parameter(
                torch.tensor(float(args.init_alpha))
            )
            self.raw_beta = torch.nn.Parameter(
                torch.tensor(float(args.init_beta))
            )

        # Final classifier consumes exactly three graph-level views.
        self.three_view_fusion = Linear(hidden * 3, hidden)
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        for module in self.modules():
            if isinstance(module, BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0.0001)

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def auxiliary_loss(self):
        losses = []
        for enhancer in self.enhancers:
            loss = enhancer.auxiliary_loss()
            if loss is not None:
                losses.append(loss)
        if not losses:
            return self.lin_class.weight.new_zeros(())
        return torch.stack(losses).mean()

    def _encode_epidemiology(self, user_state, n_hop):
        susceptible_count = torch.sum(user_state, dim=(1, 2))
        zero_count = susceptible_count.new_zeros(susceptible_count.size(0))

        U_ = self.W_u0(susceptible_count.unsqueeze(1))
        S_ = self.W_s0(zero_count.unsqueeze(1))
        D_ = self.W_d0(zero_count.unsqueeze(1))

        U_steps = []
        S_steps = []
        D_steps = []
        for _ in range(int(self.args.max_hop)):
            U_ = self.W_u(U_ - self.alpha * U_ - self.beta * U_)
            S_ = self.W_s(S_ + self.alpha * U_)
            D_ = self.W_d(D_ + self.beta * U_)
            U_steps.append(U_)
            S_steps.append(S_)
            D_steps.append(D_)

        U_hidden = torch.stack(U_steps, dim=1)
        S_hidden = torch.stack(S_steps, dim=1)
        D_hidden = torch.stack(D_steps, dim=1)

        hop_index = (
            n_hop.reshape(-1).long().clamp(1, int(self.args.max_hop)) - 1
        )
        batch_index = torch.arange(
            user_state.size(0),
            device=user_state.device,
        )
        U_last = U_hidden[batch_index, hop_index]
        S_last = S_hidden[batch_index, hop_index]
        D_last = D_hidden[batch_index, hop_index]
        epidemiology = self.W_x(
            torch.cat((U_last, S_last, D_last), dim=-1)
        )

        return (
            epidemiology,
            self.l_u(U_hidden),
            self.l_s(S_hidden),
            self.l_d(D_hidden),
        )

    def _encode_same_diff(self, data):
        x = self.bn_feat(data.x)
        x = F.relu(self.conv_feat(x, data.edge_index))
        x_diff = x

        node_state = getattr(data, "node_state", None)
        edge_stance = getattr(data, "edge_stance", None)
        for batch_norm, conv, enhancer in zip(
            self.bns_conv,
            self.convs,
            self.enhancers,
        ):
            conv_out = F.relu(conv(batch_norm(x), data.edge_index))
            same_out, diff_out = enhancer(
                conv_out,
                data.edge_index,
                node_state=node_state,
                edge_stance=edge_stance,
                return_views=True,
                tree_edge_index=data.edge_index,
                batch=data.batch,
            )
            if self.conv_residual:
                x = x + same_out
                x_diff = x_diff + diff_out
            else:
                x = same_out
                x_diff = diff_out

        same_gate = 1 if self.gating is None else self.gating(x)
        diff_gate = 1 if self.gating is None else self.gating(x_diff)
        same_graph = self.global_pool(x * same_gate, data.batch)
        diff_graph = self.global_pool(x_diff * diff_gate, data.batch)
        return same_graph, diff_graph

    def forward(self, data):
        same_graph, diff_graph = self._encode_same_diff(data)
        epidemiology, U, S, D = self._encode_epidemiology(
            data.user_state,
            data.num_hop,
        )

        x = torch.cat((same_graph, diff_graph, epidemiology), dim=-1)
        x = F.relu(self.three_view_fusion(x))

        for batch_norm, linear in zip(self.bns_fc, self.lins):
            residual = F.relu(linear(batch_norm(x)))
            x = x + residual if self.fc_residual else residual

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin_class(x)
        return F.log_softmax(logits, dim=-1), U, S, D

    def physics_loss(self, U, S, D, true_state):
        pred_states = torch.stack((U, S, D), dim=2)
        logp_state = F.log_softmax(pred_states, dim=2)

        state_sum = true_state.sum(dim=-1, keepdim=True)
        true_distribution = true_state / state_sum
        true_distribution = torch.nan_to_num(true_distribution)
        mask = (state_sum != 0).to(dtype=logp_state.dtype)
        logp_state = (mask.unsqueeze(-2) * logp_state).squeeze(-1)

        kl_div = F.kl_div(
            logp_state,
            true_distribution,
            reduction="none",
        )
        return self.args.lamda * kl_div.sum(dim=(1, 2), keepdim=True).mean()

    def __repr__(self):
        return self.__class__.__name__
