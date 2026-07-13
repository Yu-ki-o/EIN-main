"""Compact collective reinforcement and revision-response modeling.

The input stance trajectory must be root-relative. Parent-relative edge
relations are composed along each root-to-node path before this module is
called, so a denial of a denying parent is correctly treated as support for
the source claim.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _safe_logit(value):
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class CollectiveRevisionEncoder(nn.Module):
    """Encode social reinforcement and challenge-response trajectories.

    The five sequence channels are deliberately compact and inspectable:

      reinforcement, social pressure, challenge pressure,
      revision success, pressure-conditioned revision resistance.

    ``challenge`` is used instead of ``correction`` because a root-relative
    denial is observable in the conversation, while its factual
    correctness is generally unavailable in propagation-tree datasets.
    """

    def __init__(self, output_dim, args):
        super().__init__()
        self.output_dim = int(output_dim)
        self.window_k = max(
            1,
            int(getattr(args, "collective_revision_window_k", 2)),
        )
        self.uncertain_challenge_weight = max(
            0.0,
            float(
                getattr(
                    args,
                    "collective_revision_uncertain_challenge_weight",
                    0.0,
                )
            ),
        )
        self.gate_temperature = max(
            1e-4,
            float(
                getattr(
                    args,
                    "collective_revision_gate_temperature",
                    0.1,
                )
            ),
        )
        self.min_revision_gain = float(
            getattr(args, "collective_revision_min_gain", 0.05)
        )
        threshold_learnable = bool(
            getattr(
                args,
                "collective_revision_threshold_learnable",
                True,
            )
        )
        adoption_init = _safe_logit(
            getattr(
                args,
                "collective_revision_adoption_threshold_init",
                0.30,
            )
        )
        challenge_init = _safe_logit(
            getattr(
                args,
                "collective_revision_challenge_threshold_init",
                0.20,
            )
        )
        adoption = torch.tensor(adoption_init, dtype=torch.float32)
        challenge = torch.tensor(challenge_init, dtype=torch.float32)
        if threshold_learnable:
            self.raw_adoption_threshold = nn.Parameter(adoption)
            self.raw_challenge_threshold = nn.Parameter(challenge)
        else:
            self.register_buffer("raw_adoption_threshold", adoption)
            self.register_buffer("raw_challenge_threshold", challenge)

        hidden_dim = int(
            getattr(
                args,
                "collective_revision_hidden_dim",
                self.output_dim,
            )
        )
        dropout = float(getattr(args, "dropout", 0.0))
        self.sequence_encoder = nn.GRU(
            input_size=5,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Identity()
            if hidden_dim == self.output_dim
            else nn.Linear(hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.output_dim),
        )

    @property
    def adoption_threshold(self):
        return torch.sigmoid(self.raw_adoption_threshold)

    @property
    def challenge_threshold(self):
        return torch.sigmoid(self.raw_challenge_threshold)

    def _gate(self, value, threshold):
        return torch.sigmoid(
            (value - threshold) / self.gate_temperature
        )

    def _future_state(self, state, count):
        future_sum = state.new_zeros(state.size())
        future_count = count.new_zeros(count.size())
        for offset in range(1, self.window_k + 1):
            if offset >= state.size(1):
                break
            future_sum[:, :-offset] += (
                state[:, offset:] * count[:, offset:].unsqueeze(-1)
            )
            future_count[:, :-offset] += count[:, offset:]
        future_state = future_sum / future_count.clamp_min(1e-6).unsqueeze(-1)
        opportunity = future_count > 0
        return future_state, opportunity

    def forward(self, trend_sequence, num_hop):
        if trend_sequence.size(-1) < 5:
            raise ValueError(
                "collective revision requires trend channels "
                "[support, uncertain, deny, count, growth]"
            )

        state = trend_sequence[..., :3]
        support = state[..., 0]
        uncertain = state[..., 1]
        deny = state[..., 2]
        count = trend_sequence[..., 3].clamp(0.0, 1.0)
        growth = trend_sequence[..., 4]
        valid = count > 0

        # Concentration is a bounded proxy for stance homogeneity.
        concentration = (
            (state.square().sum(dim=-1) - (1.0 / 3.0))
            / (2.0 / 3.0)
        ).clamp(0.0, 1.0)
        # This is depth-wise conversational expansion, not timestamp burstiness.
        expansion = torch.tanh(F.relu(growth))
        support_majority = F.relu(support - deny)

        reinforcement = support * (
            0.4
            + 0.2 * concentration
            + 0.2 * count
            + 0.2 * expansion
        )
        reinforcement = reinforcement * valid.to(reinforcement.dtype)
        adoption_gate = self._gate(
            reinforcement,
            self.adoption_threshold,
        )

        social_pressure = (
            support_majority
            * (0.5 + 0.5 * concentration)
            * (0.5 + 0.5 * count)
            * adoption_gate
        )
        social_pressure = social_pressure * valid.to(social_pressure.dtype)

        challenge_mass = (
            deny + self.uncertain_challenge_weight * uncertain
        ).clamp(0.0, 1.0)
        challenge_pressure = challenge_mass * (0.5 + 0.5 * count)
        challenge_pressure = (
            challenge_pressure * valid.to(challenge_pressure.dtype)
        )
        challenge_gate = self._gate(
            challenge_pressure,
            self.challenge_threshold,
        )

        future_state, opportunity = self._future_state(state, count)
        opportunity = opportunity & valid
        revision_gain = 0.5 * (
            support
            - future_state[..., 0]
            + future_state[..., 2]
            - deny
        )
        success_gate = self._gate(
            revision_gain,
            revision_gain.new_tensor(self.min_revision_gain),
        )
        resistance_gate = 1.0 - success_gate
        opportunity_float = opportunity.to(dtype=state.dtype)

        revision_success = (
            opportunity_float * challenge_gate * success_gate
        )
        revision_failure = (
            opportunity_float * challenge_gate * resistance_gate
        )
        revision_resistance = (
            revision_failure * social_pressure
        )

        mechanism_sequence = torch.stack(
            (
                reinforcement,
                social_pressure,
                challenge_pressure,
                revision_success,
                revision_resistance,
            ),
            dim=-1,
        )
        mechanism_sequence = mechanism_sequence * valid.unsqueeze(-1).to(
            mechanism_sequence.dtype
        )

        hidden, _ = self.sequence_encoder(mechanism_sequence)
        last_index = (
            num_hop.view(-1).long().clamp(1, hidden.size(1)) - 1
        )
        batch_index = torch.arange(
            hidden.size(0),
            device=hidden.device,
        )
        graph = self.projection(hidden[batch_index, last_index])
        graph = graph * valid.any(dim=1, keepdim=True).to(graph.dtype)

        diagnostics = {
            "sequence": mechanism_sequence,
            "reinforcement": reinforcement,
            "social_pressure": social_pressure,
            "challenge_pressure": challenge_pressure,
            "revision_gain": revision_gain * opportunity_float,
            "revision_success": revision_success,
            "revision_failure": revision_failure,
            "revision_resistance": revision_resistance,
            "future_state": future_state,
            "opportunity": opportunity_float,
            "adoption_gate": adoption_gate * valid.to(adoption_gate.dtype),
            "challenge_gate": challenge_gate * valid.to(challenge_gate.dtype),
            "adoption_threshold": self.adoption_threshold,
            "challenge_threshold": self.challenge_threshold,
        }
        return graph, diagnostics
