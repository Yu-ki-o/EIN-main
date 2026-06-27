import math
import os
import sys
from functools import partial

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import global_add_pool, global_mean_pool

from model.EIN_ResGCN import GCNConv
from model.soft_state_uncertainty import SoftStateTargetBuilder


class UncertaintyStateSequenceEncoder(torch.nn.Module):
    """
    Encodes the uncertainty-aware soft state evolution directly.

    This replaces the EIN-style epidemiology recurrence. Instead of simulating
    U/S/D transitions from initial masses, it reads the per-depth soft state
    distribution and uncertainty produced by SoftStateTargetBuilder.
    """

    def __init__(self, hidden, max_hop, args=None):
        super().__init__()
        self.hidden = hidden
        self.max_hop = int(max_hop)
        self.uncertainty_max = max(
            1e-6,
            float(getattr(args, 'eiu_uncertainty_max', 8.0)),
        )
        self.attention_uncertainty_scale = max(
            0.0,
            float(getattr(args, 'uncertainty_attention_scale', 1.0)),
        )
        self.input_proj = torch.nn.Sequential(
            Linear(9, hidden),
            torch.nn.ReLU(),
            Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.sequence_encoder = torch.nn.GRU(
            hidden,
            hidden,
            batch_first=True,
        )
        attention_hidden = max(16, hidden // 2)
        self.depth_attention = torch.nn.Sequential(
            Linear(hidden + 1, attention_hidden),
            torch.nn.Tanh(),
            Linear(attention_hidden, 1),
        )
        self.summary_proj = torch.nn.Sequential(
            Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            Linear(hidden, hidden),
        )
        self.l_u = Linear(hidden, 1)
        self.l_s = Linear(hidden, 1)
        self.l_d = Linear(hidden, 1)
        self.l_uncertainty = Linear(hidden, 1)

    def _gather_last_hop(self, sequence, n_hop):
        hop_ind = (n_hop.clamp_min(1) - 1).long()
        hop_ind = hop_ind.view(sequence.size(0), 1, 1)
        hop_ind = hop_ind.expand(-1, -1, sequence.size(-1))
        return torch.gather(sequence, 1, hop_ind).squeeze(1)

    def forward(self, state_sequence, uncertainty_sequence, n_hop):
        batch_size, steps, _ = state_sequence.size()
        depth = torch.arange(
            1,
            steps + 1,
            device=state_sequence.device,
            dtype=state_sequence.dtype,
        ).view(1, steps, 1)
        depth = depth.expand(batch_size, -1, -1) / max(1.0, float(self.max_hop))

        uncertainty_sequence = uncertainty_sequence.clamp_min(0.0)
        uncertainty_norm = torch.log1p(
            uncertainty_sequence.clamp(max=self.uncertainty_max)
        ) / math.log1p(self.uncertainty_max)
        state_delta = torch.cat(
            (
                torch.zeros_like(state_sequence[:, :1]),
                state_sequence[:, 1:] - state_sequence[:, :-1],
            ),
            dim=1,
        )
        uncertainty_delta = torch.cat(
            (
                torch.zeros_like(uncertainty_norm[:, :1]),
                uncertainty_norm[:, 1:] - uncertainty_norm[:, :-1],
            ),
            dim=1,
        )
        features = torch.cat(
            (
                state_sequence,
                uncertainty_norm,
                depth,
                state_delta,
                uncertainty_delta,
            ),
            dim=-1,
        )
        step_hidden = self.input_proj(features)
        hidden, _ = self.sequence_encoder(step_hidden)

        valid_depth = (
            torch.arange(steps, device=state_sequence.device).view(1, steps)
            < n_hop.clamp_min(1).view(-1, 1)
        )
        attention_logits = self.depth_attention(
            torch.cat((hidden, uncertainty_norm), dim=-1)
        ).squeeze(-1)
        attention_logits = (
            attention_logits
            - self.attention_uncertainty_scale * uncertainty_norm.squeeze(-1)
        )
        attention_logits = attention_logits.masked_fill(
            ~valid_depth,
            torch.finfo(attention_logits.dtype).min,
        )
        depth_weight = F.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled_hidden = (depth_weight * hidden).sum(dim=1)
        last_hidden = self._gather_last_hop(hidden, n_hop)
        summary = self.summary_proj(torch.cat((last_hidden, pooled_hidden), dim=-1))
        summary_uncertainty = (depth_weight * uncertainty_sequence).sum(dim=1)

        U = self.l_u(hidden)
        S = self.l_s(hidden)
        D = self.l_d(hidden)
        uncertainty = F.softplus(self.l_uncertainty(hidden))
        uncertainty = uncertainty.clamp(max=self.uncertainty_max)
        return summary, summary_uncertainty, hidden, U, S, D, uncertainty


class ResGCN_Uncertainty(torch.nn.Module):
    """
    ResGCN + uncertainty-aware soft state sequence encoder.

    The original EIN-style epidemiology recurrence is replaced by a direct
    encoder over soft Support/Denial evolution and depth-wise uncertainty.
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

        self.state_sequence_encoder = UncertaintyStateSequenceEncoder(
            hidden,
            int(args.max_hop),
            args=args,
        )
        self.W_x = Linear(hidden, hidden)
        gate_hidden = max(16, hidden // 2)
        self.uncertainty_fusion_gate = torch.nn.Sequential(
            Linear(hidden * 2, gate_hidden),
            torch.nn.ReLU(),
            Linear(gate_hidden, hidden),
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
        torch.nn.init.zeros_(self.uncertainty_fusion_gate[2].weight)
        torch.nn.init.constant_(self.uncertainty_fusion_gate[2].bias, gate_bias)

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
                hidden,
                int(args.max_hop),
                args=args,
            )
        else:
            self.soft_state_builder = None

        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0.0001)

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def auxiliary_loss(self):
        if self._last_aux_loss is None:
            return self.lin_class.weight.new_zeros(())
        return self._last_aux_loss

    def _prepare_soft_state_target(self, data, h):
        self._last_soft_state_target = None
        self._last_soft_state_uncertainty = None
        self._last_aux_loss = self.lin_class.weight.new_zeros(())
        if self.soft_state_builder is None:
            return

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
        n_hop = data.num_hop
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
            n_hop,
        )
        xg = self.W_x(summary)
        self._last_uncertainty = uncertainties
        self._last_sequence_uncertainty = summary_uncertainty
        self._last_transition_probs = None
        return xg, U, S, D

    def _fuse_uncertainty(self, graph_feature, uncertainty_feature):
        gate = self.uncertainty_fusion_gate(
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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        self._prepare_soft_state_target(data, x)

        xg, U, S, D = self._uncertainty_features(data)

        for batch_norm, conv in zip(self.bns_conv, self.convs):
            x_ = batch_norm(x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_

        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = self._fuse_uncertainty(x, xg)

        for batch_norm, lin in zip(self.bns_fc, self.lins):
            x_ = batch_norm(x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
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
            return self.lin_class.weight.new_zeros(())

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
