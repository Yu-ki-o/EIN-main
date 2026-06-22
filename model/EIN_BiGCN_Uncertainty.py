import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F

from model.EIN_BiGCN import BUrumorGCN, TDrumorGCN
from model.EIN_ResGCN_Uncertainty import SoftUncertainEpidemiologyEncoder


class BiGCN_Uncertainty(torch.nn.Module):
    """
    BiGCN + EIN soft uncertain epidemiology branch.

    This mirrors the original EIN_BiGCN classifier while replacing the fixed
    alpha/beta epidemiology transition with probabilistic Support/Denial
    transfer and depth-wise uncertainty loss attenuation.
    """

    def __init__(self, in_feats, hid_feats, out_feats, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, args, device)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, args, device)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_classes)

        self.epi_encoder = SoftUncertainEpidemiologyEncoder(
            hid_feats,
            int(args.max_hop),
            args=args,
        )
        self.W_x = torch.nn.Linear(hid_feats * 4, out_feats + hid_feats)

        self._last_uncertainty = None
        self._last_transition_probs = None
        self.uncertainty_reg = max(
            0.0,
            float(getattr(args, 'eiu_uncertainty_reg', 1.0)),
        )

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _gather_last_hop(self, sequence, n_hop):
        hop_ind = (n_hop - 1).long()
        hop_ind = hop_ind.reshape(sequence.size(0), 1, 1).expand(-1, -1, sequence.size(-1))
        return torch.gather(sequence, 1, hop_ind).reshape(sequence.size(0), sequence.size(-1))

    def _epidemiology_features(self, data):
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
        ) = self.epi_encoder(data.user_state)

        U_m = self._gather_last_hop(U_hidden, data.num_hop)
        S_m = self._gather_last_hop(S_hidden, data.num_hop)
        D_m = self._gather_last_hop(D_hidden, data.num_hop)
        R_m = self._gather_last_hop(R_hidden, data.num_hop)

        self._last_uncertainty = uncertainties
        self._last_transition_probs = transition_probs
        return self.W_x(torch.cat((U_m, S_m, D_m, R_m), dim=-1)), U, S, D

    def forward(self, data):
        xg, U, S, D = self._epidemiology_features(data)

        TD_x = self.TDrumorGCN(data) + xg
        BU_x = self.BUrumorGCN(data) + xg
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1), U, S, D

    def physics_loss(self, U, S, D, true_state):
        if U is None or S is None or D is None:
            return self.fc.weight.new_zeros(())

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
