import os
import sys
from functools import partial

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import global_add_pool, global_mean_pool

from model.EIN_ResGCN import GCNConv


class SoftUncertainEpidemiologyEncoder(torch.nn.Module):
    """
    EIN epidemiology encoder with uncertainty-aware soft state transitions.

    The original EIN branch uses fixed alpha/beta rates to move hidden mass from
    Unknown to Support/Denial. Here each depth step predicts a probability over
    remain/support/denial and an uncertainty score. Support and denial therefore
    stay aligned with the original root-relative states, but their transition is
    soft and uncertainty-aware instead of deterministic.
    """

    def __init__(self, hidden, max_hop, args=None):
        super().__init__()
        self.hidden = hidden
        self.max_hop = max_hop
        self.uncertainty_max = max(
            1e-6,
            float(getattr(args, 'eiu_uncertainty_max', 8.0)),
        )

        self.W_u0 = Linear(1, hidden)
        self.W_s0 = Linear(1, hidden)
        self.W_d0 = Linear(1, hidden)

        self.W_u = Linear(hidden, hidden)
        self.W_s = Linear(hidden, hidden)
        self.W_d = Linear(hidden, hidden)
        self.W_r = Linear(1, hidden)

        self.transition_head = torch.nn.Sequential(
            Linear(hidden * 3, hidden),
            torch.nn.ReLU(),
            Linear(hidden, 3),
        )
        self.uncertainty_head = torch.nn.Sequential(
            Linear(hidden * 3, hidden),
            torch.nn.ReLU(),
            Linear(hidden, 1),
        )

        self.l_u = Linear(hidden, 1)
        self.l_s = Linear(hidden, 1)
        self.l_d = Linear(hidden, 1)

    def forward(self, user_state):
        batch_size = user_state.size(0)

        node_mass = torch.sum(user_state, dim=(1, 2)).view(batch_size, 1)
        zeros = node_mass.new_zeros(batch_size, 1)

        U_ = self.W_u0(node_mass)
        S_ = self.W_s0(zeros)
        D_ = self.W_d0(zeros)
        R_ = self.W_r(zeros)

        U_list = []
        S_list = []
        D_list = []
        R_list = []
        prob_list = []
        uncertainty_list = []

        for _ in range(self.max_hop):
            context = torch.cat((U_, S_, D_), dim=-1)
            transition_prob = F.softmax(self.transition_head(context), dim=-1)
            uncertainty = F.softplus(self.uncertainty_head(context))
            uncertainty = uncertainty.clamp(max=self.uncertainty_max)

            # High-level state remains Support/Denial relative to the root, but
            # the next layer receives probability mass instead of hard states.
            U_prev = U_
            U_ = self.W_u(transition_prob[:, 0:1] * U_prev)
            S_ = self.W_s(S_ + transition_prob[:, 1:2] * U_prev)
            D_ = self.W_d(D_ + transition_prob[:, 2:3] * U_prev)
            R_ = self.W_r(uncertainty)

            U_list.append(U_)
            S_list.append(S_)
            D_list.append(D_)
            R_list.append(R_)
            prob_list.append(transition_prob)
            uncertainty_list.append(uncertainty)

        U_hidden = torch.stack(U_list, dim=1)
        S_hidden = torch.stack(S_list, dim=1)
        D_hidden = torch.stack(D_list, dim=1)
        R_hidden = torch.stack(R_list, dim=1)
        transition_probs = torch.stack(prob_list, dim=1)
        uncertainties = torch.stack(uncertainty_list, dim=1)

        U = self.l_u(U_hidden)
        S = self.l_s(S_hidden)
        D = self.l_d(D_hidden)

        return U_hidden, S_hidden, D_hidden, R_hidden, U, S, D, transition_probs, uncertainties


class ResGCN_Uncertainty(torch.nn.Module):
    """
    ResGCN + EIN soft uncertain epidemiology branch.

    This keeps the original EIN classifier interface but replaces the fixed
    epidemiology transition with probability transfer and a depth-wise
    uncertainty signal used by the physics/KL loss.
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
        self.use_xg = True

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
        for _ in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(gconv(hidden, hidden))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, self.num_classes)

        self.epi_encoder = SoftUncertainEpidemiologyEncoder(
            hidden,
            int(args.max_hop),
            args=args,
        )
        self.W_x = Linear(hidden * 4, hidden)

        self._last_uncertainty = None
        self._last_transition_probs = None
        self.uncertainty_reg = max(
            0.0,
            float(getattr(args, 'eiu_uncertainty_reg', 1.0)),
        )

        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0.0001)

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _gather_last_hop(self, sequence, n_hop):
        hop_ind = (n_hop - 1).long()
        hop_ind = hop_ind.reshape(sequence.size(0), 1, 1).expand(-1, -1, sequence.size(-1))
        return torch.gather(sequence, 1, hop_ind).reshape(sequence.size(0), sequence.size(-1))

    def _epidemiology_features(self, data):
        user_state = data.user_state
        n_hop = data.num_hop
        (
            U_hidden,
            S_hidden,
            D_hidden,
            R_hidden,
            U,
            S,
            D,
            transition_probs,
            uncertainties,
        ) = self.epi_encoder(user_state)

        U_m = self._gather_last_hop(U_hidden, n_hop)
        S_m = self._gather_last_hop(S_hidden, n_hop)
        D_m = self._gather_last_hop(D_hidden, n_hop)
        R_m = self._gather_last_hop(R_hidden, n_hop)
        xg = self.W_x(torch.cat((U_m, S_m, D_m, R_m), dim=-1))

        self._last_uncertainty = uncertainties
        self._last_transition_probs = transition_probs
        return xg, U, S, D

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        xg, U, S, D = self._epidemiology_features(data)

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for batch_norm, conv in zip(self.bns_conv, self.convs):
            x_ = batch_norm(x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_

        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x + xg

        for batch_norm, lin in zip(self.bns_fc, self.lins):
            x_ = batch_norm(x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1), U, S, D

    def physics_loss(self, U, S, D, true_state):
        if U is None or S is None or D is None:
            return self.lin_class.weight.new_zeros(())

        pred_states = torch.cat((U, S, D), dim=-1)
        logp_state = F.log_softmax(pred_states, dim=-1)

        target_mass = true_state.sum(dim=-1, keepdim=True)
        mask = target_mass > 0
        target = true_state / target_mass.clamp_min(1e-6)
        target = torch.where(mask, target, torch.zeros_like(target))

        kl_div = F.kl_div(logp_state, target, reduction='none').sum(dim=-1, keepdim=True)

        if self._last_uncertainty is None:
            uncertainty = torch.zeros_like(kl_div)
        else:
            uncertainty = self._last_uncertainty[:, :kl_div.size(1), :]
            if uncertainty.size(1) < kl_div.size(1):
                pad = kl_div.new_zeros(kl_div.size(0), kl_div.size(1) - uncertainty.size(1), 1)
                uncertainty = torch.cat((uncertainty, pad), dim=1)

        attenuated = torch.exp(-uncertainty) * kl_div + self.uncertainty_reg * uncertainty
        attenuated = attenuated * mask.to(dtype=attenuated.dtype)
        denom = mask.to(dtype=attenuated.dtype).sum().clamp_min(1.0)
        return self.args.lamda * attenuated.sum() / denom

    def __repr__(self):
        return self.__class__.__name__
