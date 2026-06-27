import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F

from model.EIN_BiGCN import BUrumorGCN, TDrumorGCN
from model.EIN_ResGCN_Uncertainty import UncertaintyStateSequenceEncoder
from model.soft_state_uncertainty import SoftStateTargetBuilder


class BiGCN_Uncertainty(torch.nn.Module):
    """
    BiGCN + uncertainty-aware soft state sequence encoder.

    The TD/BU graph backbone is kept, while the original EIN-style recurrence is
    replaced by direct encoding of soft Support/Denial evolution and uncertainty.
    """

    def __init__(self, in_feats, hid_feats, out_feats, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, args, device)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, args, device)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_classes)
        self.state_feat = torch.nn.Linear(in_feats, hid_feats)

        self.state_sequence_encoder = UncertaintyStateSequenceEncoder(
            hid_feats,
            int(args.max_hop),
            args=args,
        )
        branch_dim = out_feats + hid_feats
        self.W_x = torch.nn.Linear(hid_feats, branch_dim)
        gate_hidden = max(16, branch_dim // 2)
        self.td_uncertainty_gate = torch.nn.Sequential(
            torch.nn.Linear(branch_dim * 2, gate_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(gate_hidden, branch_dim),
            torch.nn.Sigmoid(),
        )
        self.bu_uncertainty_gate = torch.nn.Sequential(
            torch.nn.Linear(branch_dim * 2, gate_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(gate_hidden, branch_dim),
            torch.nn.Sigmoid(),
        )
        self.fusion_confidence_scale = max(
            0.0,
            float(getattr(args, 'uncertainty_fusion_confidence_scale', 0.5)),
        )
        self.fusion_confidence_floor = min(
            max(
                float(getattr(args, 'uncertainty_fusion_confidence_floor', 0.25)),
                0.0,
            ),
            1.0,
        )
        gate_bias = float(getattr(args, 'uncertainty_fusion_gate_bias', -2.0))
        for gate in (self.td_uncertainty_gate, self.bu_uncertainty_gate):
            torch.nn.init.zeros_(gate[2].weight)
            torch.nn.init.constant_(gate[2].bias, gate_bias)

        self._last_uncertainty = None
        self._last_sequence_uncertainty = None
        self._last_transition_probs = None
        self._last_soft_state_target = None
        self._last_soft_state_uncertainty = None
        self._last_aux_loss = None
        self.uncertainty_reg = max(
            0.0,
            float(getattr(args, 'eiu_uncertainty_reg', 1.0)),
        )
        self.use_soft_state_targets = bool(
            getattr(args, 'use_soft_state_targets', True)
        )
        self.soft_state_uncertainty_weight = max(
            0.0,
            float(getattr(args, 'soft_state_uncertainty_weight', 1.0)),
        )
        self.state_loss_weight_floor = max(
            0.0,
            float(getattr(args, 'state_loss_weight_floor', 0.2)),
        )
        if self.use_soft_state_targets:
            self.soft_state_builder = SoftStateTargetBuilder(
                hid_feats,
                int(args.max_hop),
                args=args,
            )
        else:
            self.soft_state_builder = None

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def auxiliary_loss(self):
        if self._last_aux_loss is None:
            return self.fc.weight.new_zeros(())
        return self._last_aux_loss

    def _prepare_soft_state_target(self, data):
        self._last_soft_state_target = None
        self._last_soft_state_uncertainty = None
        self._last_aux_loss = self.fc.weight.new_zeros(())
        if self.soft_state_builder is None:
            return

        h = F.relu(self.state_feat(data.x.float()))
        target, uncertainty, edge_loss = self.soft_state_builder(data, h)
        self._last_soft_state_target = target.detach()
        self._last_soft_state_uncertainty = uncertainty.detach()
        self._last_aux_loss = edge_loss

    def _state_sequence_source(self, data):
        if self._last_soft_state_target is not None:
            return self._last_soft_state_target, self._last_soft_state_uncertainty

        target_mass = data.user_state.sum(dim=-1, keepdim=True)
        mask = target_mass > 0
        state_sequence = data.user_state / target_mass.clamp_min(1e-6)
        state_sequence = torch.where(mask, state_sequence, torch.zeros_like(state_sequence))
        uncertainty = state_sequence.new_zeros(state_sequence.size(0), state_sequence.size(1), 1)
        return state_sequence, uncertainty

    def _uncertainty_features(self, data):
        state_sequence, uncertainty_sequence = self._state_sequence_source(data)
        (
            summary,
            summary_uncertainty,
            _,
            U,
            S,
            D,
            uncertainties,
        ) = self.state_sequence_encoder(
            state_sequence,
            uncertainty_sequence,
            data.num_hop,
        )
        self._last_uncertainty = uncertainties
        self._last_sequence_uncertainty = summary_uncertainty
        self._last_transition_probs = None
        xg = self.W_x(summary)
        return xg, U, S, D

    def _fuse_uncertainty(self, graph_feature, uncertainty_feature, gate_module):
        gate = gate_module(
            torch.cat((graph_feature, uncertainty_feature), dim=-1)
        )
        if self._last_sequence_uncertainty is None:
            confidence = graph_feature.new_ones(graph_feature.size(0), 1)
        else:
            confidence = torch.exp(
                -self.fusion_confidence_scale * self._last_sequence_uncertainty
            ).clamp_min(self.fusion_confidence_floor)
        return graph_feature + gate * confidence * uncertainty_feature

    def forward(self, data):
        self._prepare_soft_state_target(data)
        xg, U, S, D = self._uncertainty_features(data)

        TD_x = self._fuse_uncertainty(
            self.TDrumorGCN(data),
            xg,
            self.td_uncertainty_gate,
        )
        BU_x = self._fuse_uncertainty(
            self.BUrumorGCN(data),
            xg,
            self.bu_uncertainty_gate,
        )
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1), U, S, D

    def _normalize_target(self, target, mask=None):
        target_mass = target.sum(dim=-1, keepdim=True)
        if mask is None:
            mask = target_mass > 0
        normalized = target / target_mass.clamp_min(1e-6)
        return torch.where(mask, normalized, torch.zeros_like(normalized)), mask

    def _physics_target(self, true_state):
        if self._last_soft_state_target is not None:
            soft_target, soft_mask = self._normalize_target(self._last_soft_state_target)
            return soft_target, soft_mask
        return self._normalize_target(true_state)

    def _match_depth_tensor(self, sequence, reference):
        if sequence is None:
            return torch.zeros_like(reference)
        sequence = sequence[:, :reference.size(1), :]
        if sequence.size(1) < reference.size(1):
            pad = reference.new_zeros(
                reference.size(0),
                reference.size(1) - sequence.size(1),
                1,
            )
            sequence = torch.cat((sequence, pad), dim=1)
        return sequence

    def _uncertainty_terms(self, kl_div):
        predicted = self._match_depth_tensor(self._last_uncertainty, kl_div)
        if self._last_soft_state_uncertainty is None:
            return predicted, predicted

        evidence = self._match_depth_tensor(
            self._last_soft_state_uncertainty,
            kl_div,
        )
        evidence = self.soft_state_uncertainty_weight * evidence
        calibration = F.smooth_l1_loss(
            predicted,
            evidence.detach(),
            reduction='none',
        )
        return evidence, calibration

    def physics_loss(self, U, S, D, true_state):
        if U is None or S is None or D is None:
            return self.fc.weight.new_zeros(())

        pred_states = torch.cat((U, S, D), dim=-1)
        logp_state = F.log_softmax(pred_states, dim=-1)

        target, mask = self._physics_target(true_state)
        kl_div = F.kl_div(logp_state, target, reduction='none').sum(dim=-1, keepdim=True)
        uncertainty, uncertainty_penalty = self._uncertainty_terms(kl_div)

        state_weight = torch.exp(-uncertainty).clamp_min(self.state_loss_weight_floor)
        attenuated = state_weight * kl_div + self.uncertainty_reg * uncertainty_penalty
        attenuated = attenuated * mask.to(dtype=attenuated.dtype)
        denom = mask.to(dtype=attenuated.dtype).sum().clamp_min(1.0)
        return self.args.lamda * attenuated.sum() / denom

    def __repr__(self):
        return self.__class__.__name__
