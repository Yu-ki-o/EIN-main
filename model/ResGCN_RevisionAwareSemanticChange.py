import math

import torch
from torch import nn
import torch.nn.functional as F

from model.ResGCN_UncertaintySemanticChange import (
    ResGCN_UncertaintySemanticChange,
)


def _safe_logit(value):
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class ResGCN_RevisionAwareSemanticChange(
    ResGCN_UncertaintySemanticChange
):
    """
    ResGCN backbone with a compact correction-resistant revision signal.

    The mechanism is intentionally folded into the existing trend branch:
      support, uncertain, deny, count, growth, revision_anomaly
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
        super().__init__(
            in_feats=in_feats,
            hid_feats=hid_feats,
            out_feats=out_feats,
            num_classes=num_classes,
            args=args,
            device=device,
        )
        self.use_revision_mechanism = bool(
            getattr(args, "use_revision_mechanism", True)
        )
        self.revision_uncertain_weight = float(
            getattr(args, "revision_uncertain_weight", 0.5)
        )
        self.revision_gate_temperature = max(
            1e-4,
            float(getattr(args, "revision_gate_temperature", 0.2)),
        )
        self.revision_use_support_pressure = bool(
            getattr(args, "revision_use_support_pressure", True)
        )
        self.revision_threshold_learnable = bool(
            getattr(args, "revision_threshold_learnable", True)
        )

        challenge_init = float(
            getattr(args, "revision_challenge_threshold_init", 0.15)
        )
        pressure_init = float(
            getattr(args, "revision_pressure_threshold_init", 0.15)
        )
        challenge = torch.tensor(_safe_logit(challenge_init))
        pressure = torch.tensor(_safe_logit(pressure_init))
        if self.revision_threshold_learnable:
            self.raw_revision_challenge_threshold = nn.Parameter(challenge)
            self.raw_revision_pressure_threshold = nn.Parameter(pressure)
        else:
            self.register_buffer(
                "raw_revision_challenge_threshold",
                challenge,
            )
            self.register_buffer(
                "raw_revision_pressure_threshold",
                pressure,
            )

        if self.use_revision_mechanism:
            trend_hidden = self.uncertainty_trend_encoder.hidden_size
            self.uncertainty_trend_encoder = nn.GRU(
                input_size=6,
                hidden_size=trend_hidden,
                batch_first=True,
            )
        self._last_revision_anomaly_sequence = None

    @property
    def revision_challenge_threshold(self):
        return torch.sigmoid(self.raw_revision_challenge_threshold)

    @property
    def revision_pressure_threshold(self):
        return torch.sigmoid(self.raw_revision_pressure_threshold)

    def _gate(self, value, threshold):
        return torch.sigmoid(
            (value - threshold) / self.revision_gate_temperature
        )

    def _build_revision_anomaly(self, base_trend):
        support = base_trend[..., 0:1]
        uncertain = base_trend[..., 1:2]
        deny = base_trend[..., 2:3]
        count = base_trend[..., 3:4]

        next_support = torch.cat(
            (
                support[:, 1:],
                support.new_zeros(support.size(0), 1, 1),
            ),
            dim=1,
        )
        support_increase = F.relu(next_support - support)
        challenge_pressure = (
            deny + self.revision_uncertain_weight * uncertain
        ) * count
        support_pressure = F.relu(support - deny) * count

        challenge_gate = self._gate(
            challenge_pressure,
            self.revision_challenge_threshold,
        )
        if self.revision_use_support_pressure:
            pressure_gate = self._gate(
                support_pressure,
                self.revision_pressure_threshold,
            )
        else:
            pressure_gate = torch.ones_like(challenge_gate)

        anomaly = challenge_gate * pressure_gate * support_increase
        return anomaly.clamp(0.0, 1.0)

    def _build_uncertainty_trend(
        self,
        data,
        probabilities,
        keep_sample,
    ):
        base_trend = super()._build_uncertainty_trend(
            data,
            probabilities,
            keep_sample,
        )
        if not self.use_revision_mechanism:
            self._last_revision_anomaly_sequence = None
            return base_trend

        anomaly = self._build_revision_anomaly(base_trend)
        self._last_revision_anomaly_sequence = anomaly.detach()
        return torch.cat((base_trend, anomaly), dim=-1)
