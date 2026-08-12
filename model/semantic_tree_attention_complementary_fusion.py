import torch
import torch.nn.functional as F
from torch import nn


class SemanticTreeAttentionComplementaryFusion(nn.Module):
    """Label-sufficient, conditionally complementary branch fusion.

    The module keeps the existing change/Semantic-tree classifier path as its
    base and learns only a zero-initialized residual correction.  Semantic-tree
    attention probabilities select high- and low-attention node evidence.  The
    former is supervised to be label sufficient and conditionally decorrelated
    from the change graph representation; the latter provides a within-graph
    ranking counterfactual during training.
    """

    def __init__(self, hidden_dim, num_classes, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temperature = max(
            1e-4,
            float(
                getattr(
                    args,
                    "semantic_tree_evidence_temperature",
                    0.5,
                )
            ),
        )
        self.rank_margin = max(
            0.0,
            float(
                getattr(
                    args,
                    "semantic_tree_evidence_rank_margin",
                    0.2,
                )
            ),
        )
        self.warmup_epochs = max(
            0,
            int(
                getattr(
                    args,
                    "semantic_tree_evidence_warmup_epochs",
                    5,
                )
            ),
        )
        self.stop_change_gradient = bool(
            getattr(
                args,
                "semantic_tree_conditional_redundancy_stop_change",
                True,
            )
        )
        interaction_hidden = max(
            self.hidden_dim,
            int(
                getattr(
                    args,
                    "semantic_tree_evidence_interaction_hidden_dim",
                    self.hidden_dim * 2,
                )
            ),
        )
        interaction_dropout = float(
            getattr(
                args,
                "semantic_tree_evidence_interaction_dropout",
                getattr(args, "dropout", 0.0),
            )
        )
        residual_init = float(
            getattr(
                args,
                "semantic_tree_evidence_residual_init",
                0.0,
            )
        )
        self.eps = 1e-6
        self.register_buffer(
            "_current_epoch",
            torch.zeros((), dtype=torch.long),
        )

        self.evidence_classifier = nn.Linear(
            self.hidden_dim,
            int(num_classes),
        )
        self.interaction_encoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, interaction_hidden),
            nn.GELU(),
            nn.Dropout(interaction_dropout),
            nn.Linear(interaction_hidden, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(residual_init, dtype=torch.float32)
        )

        self.last_high_weight = None
        self.last_low_weight = None
        self.last_high_evidence = None
        self.last_low_evidence = None
        self.last_high_logits = None
        self.last_low_logits = None
        self.last_interaction = None
        self.last_effective_residual_scale = None
        self.last_sufficiency_loss = None
        self.last_rank_loss = None
        self.last_conditional_redundancy_loss = None

    @property
    def current_epoch(self):
        return int(self._current_epoch.item())

    def set_epoch(self, epoch):
        self._current_epoch.fill_(max(0, int(epoch)))

    def auxiliary_ramp(self):
        if not self.training or self.warmup_epochs <= 0:
            return 1.0
        return min(
            1.0,
            float(self.current_epoch + 1) / float(self.warmup_epochs),
        )

    def _evidence_weights(self, attention_probability, valid_mask):
        if attention_probability.dim() != 3:
            raise ValueError(
                "Semantic-tree attention probability must have shape "
                "[batch, queries, nodes]"
            )
        if valid_mask.dim() != 2:
            raise ValueError(
                "Semantic-tree valid mask must have shape [batch, nodes]"
            )
        attention = attention_probability.mean(dim=1)
        attention = attention.clamp_min(self.eps)
        log_attention = attention.log()
        invalid = ~valid_mask.bool()
        high_logits = (log_attention / self.temperature).masked_fill(
            invalid,
            -1e9,
        )
        low_logits = (-log_attention / self.temperature).masked_fill(
            invalid,
            -1e9,
        )
        return F.softmax(high_logits, dim=-1), F.softmax(
            low_logits,
            dim=-1,
        )

    def _conditional_redundancy(self, high_evidence, change_graph, target):
        zero = high_evidence.new_zeros(())
        if target is None or high_evidence.size(0) < 2:
            return zero
        target = target.view(-1).long()
        change = (
            change_graph.detach()
            if self.stop_change_gradient
            else change_graph
        )
        high = F.layer_norm(high_evidence, (self.hidden_dim,))
        change = F.layer_norm(change, (self.hidden_dim,))
        class_losses = []
        for class_id in target.unique(sorted=True):
            class_mask = target.eq(class_id)
            sample_count = int(class_mask.sum().item())
            if sample_count < 2:
                continue
            class_high = high[class_mask]
            class_change = change[class_mask]
            class_high = class_high - class_high.mean(dim=0, keepdim=True)
            class_change = class_change - class_change.mean(
                dim=0,
                keepdim=True,
            )
            cross_correlation = class_high.t().matmul(class_change)
            cross_correlation = cross_correlation / float(sample_count - 1)
            class_losses.append(cross_correlation.pow(2).mean())
        if not class_losses:
            return zero
        return torch.stack(class_losses).mean()

    def forward(
        self,
        base_fused,
        change_graph,
        value_dense,
        attention_probability,
        valid_mask,
        target=None,
    ):
        if value_dense.dim() != 3:
            raise ValueError(
                "Semantic-tree values must have shape [batch, nodes, hidden]"
            )
        if value_dense.size(-1) != self.hidden_dim:
            raise ValueError(
                "Semantic-tree value dimension does not match fusion hidden "
                "dimension"
            )
        high_weight, low_weight = self._evidence_weights(
            attention_probability,
            valid_mask,
        )
        high_evidence = torch.bmm(
            high_weight.unsqueeze(1),
            value_dense,
        ).squeeze(1)
        low_evidence = torch.bmm(
            low_weight.unsqueeze(1),
            value_dense,
        ).squeeze(1)

        interaction_input = torch.cat(
            (
                change_graph,
                high_evidence,
                change_graph * high_evidence,
                (change_graph - high_evidence).abs(),
            ),
            dim=-1,
        )
        interaction = self.interaction_encoder(interaction_input)
        effective_scale = torch.tanh(self.residual_scale)
        fused = base_fused + effective_scale * interaction

        high_logits = self.evidence_classifier(high_evidence)
        low_logits = None
        sufficiency_loss = None
        rank_loss = None
        redundancy_loss = None
        if target is not None:
            target = target.view(-1).long()
            high_log_probability = F.log_softmax(high_logits, dim=-1)
            sufficiency_loss = F.nll_loss(high_log_probability, target)
            low_logits = self.evidence_classifier(low_evidence)
            high_correct = high_logits.gather(
                1,
                target.unsqueeze(1),
            ).squeeze(1)
            low_correct = low_logits.gather(
                1,
                target.unsqueeze(1),
            ).squeeze(1)
            eligible = valid_mask.sum(dim=-1).gt(1)
            if eligible.any():
                rank_loss = F.relu(
                    self.rank_margin
                    - high_correct[eligible]
                    + low_correct[eligible]
                ).mean()
            else:
                rank_loss = high_evidence.new_zeros(())
            redundancy_loss = self._conditional_redundancy(
                high_evidence,
                change_graph,
                target,
            )

        self.last_high_weight = high_weight
        self.last_low_weight = low_weight
        self.last_high_evidence = high_evidence
        self.last_low_evidence = low_evidence
        self.last_high_logits = high_logits
        self.last_low_logits = low_logits
        self.last_interaction = interaction
        self.last_effective_residual_scale = effective_scale
        self.last_sufficiency_loss = sufficiency_loss
        self.last_rank_loss = rank_loss
        self.last_conditional_redundancy_loss = redundancy_loss
        return fused
