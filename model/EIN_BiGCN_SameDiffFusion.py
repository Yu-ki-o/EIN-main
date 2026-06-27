import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F

from model.BiGCN_StateAuxSameDiff import (
    BUrumorGCN_StateAuxSameDiff,
    TDrumorGCN_StateAuxSameDiff,
)


class EINBiGCNSameDiffFusion(torch.nn.Module):
    """
    EIN BiGCN with same, diff, and original U/S/D epidemiology views.

    The bidirectional same and diff graph representations remain independent
    until they are fused with the original EIN epidemiology representation for
    final classification.
    """

    def __init__(
        self,
        in_feats,
        hid_feats,
        out_feats,
        num_classes,
        args,
        device,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden = hid_feats
        self.dropout = float(getattr(args, "dropout", 0.5))
        if not bool(getattr(args, "state_aux_dual_view", True)):
            raise ValueError(
                "EINBiGCNSameDiffFusion requires state_aux_dual_view=True"
            )

        self.TDrumorGCN = TDrumorGCN_StateAuxSameDiff(
            in_feats,
            hid_feats,
            out_feats,
            args,
            device,
        )
        self.BUrumorGCN = BUrumorGCN_StateAuxSameDiff(
            in_feats,
            hid_feats,
            out_feats,
            args,
            device,
        )

        # Original EIN epidemiology encoder.
        self.W_u0 = torch.nn.Linear(1, hid_feats)
        self.W_s0 = torch.nn.Linear(1, hid_feats)
        self.W_d0 = torch.nn.Linear(1, hid_feats)
        self.W_u = torch.nn.Linear(hid_feats, hid_feats)
        self.W_s = torch.nn.Linear(hid_feats, hid_feats)
        self.W_d = torch.nn.Linear(hid_feats, hid_feats)
        branch_dim = out_feats + hid_feats
        self.W_x = torch.nn.Linear(hid_feats * 3, branch_dim)
        self.l_u = torch.nn.Linear(hid_feats, 1)
        self.l_s = torch.nn.Linear(hid_feats, 1)
        self.l_d = torch.nn.Linear(hid_feats, 1)

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

        graph_dim = branch_dim * 2
        self.three_view_fusion = torch.nn.Linear(graph_dim * 3, graph_dim)
        self.fc = torch.nn.Linear(graph_dim, num_classes)

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
        for module in (self.TDrumorGCN, self.BUrumorGCN):
            loss = module.auxiliary_loss()
            if loss is not None:
                losses.append(loss)
        if not losses:
            return self.fc.weight.new_zeros(())
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
        epidemiology_branch = self.W_x(
            torch.cat((U_last, S_last, D_last), dim=-1)
        )

        # Original EIN adds the same epidemiology vector to TD and BU.
        epidemiology_graph = torch.cat(
            (epidemiology_branch, epidemiology_branch),
            dim=-1,
        )
        return (
            epidemiology_graph,
            self.l_u(U_hidden),
            self.l_s(S_hidden),
            self.l_d(D_hidden),
        )

    def forward(self, data):
        td_same, td_diff = self.TDrumorGCN(data)
        bu_same, bu_diff = self.BUrumorGCN(data)

        same_graph = torch.cat((bu_same, td_same), dim=-1)
        diff_graph = torch.cat((bu_diff, td_diff), dim=-1)
        epidemiology_graph, U, S, D = self._encode_epidemiology(
            data.user_state,
            data.num_hop,
        )

        x = torch.cat(
            (same_graph, diff_graph, epidemiology_graph),
            dim=-1,
        )
        x = F.relu(self.three_view_fusion(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc(x)
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
