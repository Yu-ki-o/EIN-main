import math

import torch
import torch.nn.functional as F
from torch import nn


class TrajectoryPrototypeChangeEnhancer(nn.Module):
    """Enhance the Change branch with multi-hop trajectory prototypes.

    The input sequence is formed by the intermediate support/deny states of
    the semantic parity encoder.  Its index denotes propagation depth (and
    therefore an enlarged receptive field), not wall-clock time.

    The module first builds a shared support/deny interaction state at every
    layer.  Consecutive states are then encoded as a transition trajectory and
    softly matched against learnable, stage-aware prototypes.  The individual
    trajectory and its matched prototype prior generate residual corrections
    for the final support/deny states consumed by the existing Change encoder.
    """

    def __init__(self, hidden_dim, num_layers, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers < 2:
            raise ValueError(
                "trajectory prototypes require at least two parity layers"
            )

        self.num_transitions = self.num_layers - 1
        self.num_prototypes = max(
            2,
            int(getattr(args, "trajectory_prototype_num", 4)),
        )
        self.pattern_dim = max(
            1,
            int(
                getattr(
                    args,
                    "trajectory_prototype_dim",
                    self.hidden_dim,
                )
            ),
        )
        self.temperature = max(
            1e-4,
            float(
                getattr(
                    args,
                    "trajectory_prototype_temperature",
                    0.5,
                )
            ),
        )
        self.dropout = max(
            0.0,
            float(getattr(args, "trajectory_prototype_dropout", 0.0)),
        )
        self.lambda_diversity = max(
            0.0,
            float(
                getattr(
                    args,
                    "lambda_trajectory_prototype_diversity_aux",
                    0.001,
                )
            ),
        )
        self.lambda_balance = max(
            0.0,
            float(
                getattr(
                    args,
                    "lambda_trajectory_prototype_balance_aux",
                    0.001,
                )
            ),
        )

        # Shared across propagation layers so their interaction states live in
        # a directly comparable semantic space.
        self.interaction_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.pattern_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pattern_dim, self.pattern_dim, bias=False),
            nn.LayerNorm(self.pattern_dim),
        )
        # A transition retains its endpoints, signed change, and persistence.
        self.transition_encoder = nn.Sequential(
            nn.Linear(self.pattern_dim * 4, self.pattern_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pattern_dim, self.pattern_dim, bias=False),
            nn.LayerNorm(self.pattern_dim),
        )
        self.stage_score = nn.Linear(self.pattern_dim, 1, bias=False)
        self.pattern_fusion = nn.Sequential(
            nn.Linear(self.pattern_dim * 3, self.pattern_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.residual_projection = nn.Linear(
            self.pattern_dim,
            self.hidden_dim * 2,
            bias=False,
        )

        self.prototypes = nn.Parameter(
            torch.empty(
                self.num_prototypes,
                self.num_transitions,
                self.pattern_dim,
            )
        )
        nn.init.normal_(
            self.prototypes,
            mean=0.0,
            std=1.0 / math.sqrt(self.pattern_dim),
        )
        gate_init = float(
            getattr(
                args,
                "trajectory_prototype_residual_gate_init",
                -2.0,
            )
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(gate_init))

        self.last_outputs = None

    def _validate_layers(self, support_layers, deny_layers):
        if len(support_layers) != self.num_layers:
            raise ValueError(
                "expected {} support layers, got {}".format(
                    self.num_layers,
                    len(support_layers),
                )
            )
        if len(deny_layers) != self.num_layers:
            raise ValueError(
                "expected {} deny layers, got {}".format(
                    self.num_layers,
                    len(deny_layers),
                )
            )
        reference_shape = support_layers[0].shape
        for support, deny in zip(support_layers, deny_layers):
            if support.shape != reference_shape or deny.shape != reference_shape:
                raise ValueError(
                    "all trajectory layers must have identical shapes"
                )
            if support.size(-1) != self.hidden_dim:
                raise ValueError(
                    "expected hidden dimension {}, got {}".format(
                        self.hidden_dim,
                        support.size(-1),
                    )
                )

    def _interaction_state(self, support, deny):
        return self.interaction_encoder(
            torch.cat(
                (
                    support,
                    deny,
                    support - deny,
                    support * deny,
                ),
                dim=-1,
            )
        )

    def _transition_state(self, previous, current):
        return self.transition_encoder(
            torch.cat(
                (
                    previous,
                    current,
                    current - previous,
                    current * previous,
                ),
                dim=-1,
            )
        )

    def _prototype_losses(self, assignment, node_weight=None):
        flattened = self.prototypes.flatten(start_dim=1)
        normalized = F.normalize(flattened, p=2, dim=-1, eps=1e-8)
        gram = normalized @ normalized.transpose(0, 1)
        identity = torch.eye(
            self.num_prototypes,
            dtype=gram.dtype,
            device=gram.device,
        )
        diversity = (gram - identity).pow(2).mean()

        if assignment.size(0) == 0:
            balance = diversity.new_zeros(())
        else:
            if node_weight is None:
                average_assignment = assignment.mean(dim=0)
            else:
                weight = node_weight.to(
                    device=assignment.device,
                    dtype=assignment.dtype,
                ).view(-1, 1).clamp_min(0.0)
                average_assignment = (
                    (assignment * weight).sum(dim=0)
                    / weight.sum().clamp_min(1e-8)
                )
            uniform_log_probability = -math.log(self.num_prototypes)
            balance = (
                average_assignment.clamp_min(1e-8)
                * (
                    average_assignment.clamp_min(1e-8).log()
                    - uniform_log_probability
                )
            ).sum()
        return diversity, balance

    def forward(
        self,
        support_layers,
        deny_layers,
        node_weight=None,
    ):
        support_layers = tuple(support_layers)
        deny_layers = tuple(deny_layers)
        self._validate_layers(support_layers, deny_layers)

        interactions = torch.stack(
            [
                self._interaction_state(support, deny)
                for support, deny in zip(support_layers, deny_layers)
            ],
            dim=1,
        )
        transitions = torch.stack(
            [
                self._transition_state(
                    interactions[:, index - 1],
                    interactions[:, index],
                )
                for index in range(1, self.num_layers)
            ],
            dim=1,
        )

        transition_normalized = F.normalize(
            transitions,
            p=2,
            dim=-1,
            eps=1e-8,
        )
        prototype_normalized = F.normalize(
            self.prototypes,
            p=2,
            dim=-1,
            eps=1e-8,
        )
        stage_similarity = torch.einsum(
            "ntd,ktd->nkt",
            transition_normalized,
            prototype_normalized,
        )
        similarity = stage_similarity.mean(dim=-1) / self.temperature
        assignment = F.softmax(similarity, dim=-1)

        matched_trajectory = torch.einsum(
            "nk,ktd->ntd",
            assignment,
            self.prototypes,
        )
        stage_attention = F.softmax(
            self.stage_score(transitions).squeeze(-1),
            dim=-1,
        )
        trajectory_summary = torch.sum(
            stage_attention.unsqueeze(-1) * transitions,
            dim=1,
        )
        prototype_summary = torch.sum(
            stage_attention.unsqueeze(-1) * matched_trajectory,
            dim=1,
        )
        pattern_hidden = self.pattern_fusion(
            torch.cat(
                (
                    interactions[:, -1],
                    trajectory_summary,
                    prototype_summary,
                ),
                dim=-1,
            )
        )
        support_residual, deny_residual = self.residual_projection(
            pattern_hidden
        ).chunk(2, dim=-1)
        residual_gate = torch.sigmoid(self.residual_gate_logit)
        enhanced_support = (
            support_layers[-1] + residual_gate * support_residual
        )
        enhanced_deny = deny_layers[-1] + residual_gate * deny_residual

        diversity_loss, balance_loss = self._prototype_losses(
            assignment,
            node_weight=node_weight,
        )
        aux_loss = (
            self.lambda_diversity * diversity_loss
            + self.lambda_balance * balance_loss
        )
        outputs = {
            "support": enhanced_support,
            "deny": enhanced_deny,
            "interactions": interactions,
            "transitions": transitions,
            "stage_similarity": stage_similarity,
            "similarity": similarity,
            "assignment": assignment,
            "stage_attention": stage_attention,
            "trajectory_summary": trajectory_summary,
            "matched_trajectory": matched_trajectory,
            "prototype_summary": prototype_summary,
            "residual_gate": residual_gate,
            "aux_loss": aux_loss,
            "diversity_loss": diversity_loss,
            "balance_loss": balance_loss,
        }
        self.last_outputs = outputs
        return enhanced_support, enhanced_deny, outputs
