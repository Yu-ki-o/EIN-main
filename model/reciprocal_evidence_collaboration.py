import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch


class ReciprocalEvidenceCollaboration(nn.Module):
    """Change proposal -> tree verification -> counterfactual feedback.

    The module deliberately keeps the two base graph branches intact.  It
    builds graph-conditioned evidence queries from node-level change features,
    lets the semantic-tree branch verify those queries, and then injects the
    verified evidence into the two graph representations through gated
    residuals.  Counterfactual leave-one-slot-out supervision is used only
    while training.
    """

    def __init__(self, hidden_dim, num_classes, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_slots = max(
            1,
            int(getattr(args, "repv_num_slots", 4)),
        )
        self.proposal_temperature = max(
            1e-6,
            float(getattr(args, "repv_proposal_temperature", 1.0)),
        )
        self.proposal_bias_scale = max(
            0.0,
            float(getattr(args, "repv_proposal_bias_scale", 0.5)),
        )
        self.counterfactual_temperature = max(
            1e-6,
            float(
                getattr(args, "repv_counterfactual_temperature", 1.0)
            ),
        )
        self.warmup_epochs = max(
            0,
            int(getattr(args, "repv_warmup_epochs", 5)),
        )
        self.ramp_epochs = max(
            1,
            int(getattr(args, "repv_ramp_epochs", 5)),
        )
        self.lambda_classification = max(
            0.0,
            float(getattr(args, "lambda_repv_classification_aux", 0.1)),
        )
        self.lambda_feedback = max(
            0.0,
            float(getattr(args, "lambda_repv_feedback_aux", 0.1)),
        )
        self.lambda_overlap = max(
            0.0,
            float(getattr(args, "lambda_repv_overlap_aux", 0.01)),
        )
        self.lambda_diversity = max(
            0.0,
            float(getattr(args, "lambda_repv_diversity_aux", 0.001)),
        )
        self.eps = max(
            1e-12,
            float(getattr(args, "repv_eps", 1e-6)),
        )
        dropout = float(
            getattr(args, "repv_dropout", getattr(args, "dropout", 0.0))
        )

        self.slot_queries = nn.Parameter(
            torch.empty(self.num_slots, self.hidden_dim)
        )
        self.tree_query_base = nn.Parameter(
            torch.empty(self.num_slots, self.hidden_dim)
        )
        nn.init.normal_(self.slot_queries, std=self.hidden_dim ** -0.5)
        nn.init.normal_(self.tree_query_base, std=self.hidden_dim ** -0.5)

        self.change_key = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.change_value = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.GELU(),
        )
        self.prototype_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.root_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)

        self.slot_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        self.slot_importance = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.collaboration_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.collaboration_classifier = nn.Linear(
            self.hidden_dim,
            self.num_classes,
        )
        self.change_residual = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.tree_residual = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        gate_init = float(getattr(args, "repv_residual_gate_init", -2.0))
        self.change_gate_logit = nn.Parameter(torch.tensor(gate_init))
        self.tree_gate_logit = nn.Parameter(torch.tensor(gate_init))

        self.register_buffer(
            "current_epoch",
            torch.zeros((), dtype=torch.long),
        )

    def set_epoch(self, epoch):
        self.current_epoch.fill_(max(0, int(epoch)))

    def feedback_scale(self):
        if not self.training:
            return 1.0
        epoch = int(self.current_epoch.item())
        if epoch < self.warmup_epochs:
            return 0.0
        return min(
            1.0,
            float(epoch - self.warmup_epochs + 1) / self.ramp_epochs,
        )

    def propose(self, change_nodes, root_nodes, batch):
        if change_nodes is None or root_nodes is None:
            raise ValueError(
                "REPV requires both change_nodes and root_nodes"
            )
        change_dense, valid_mask = to_dense_batch(change_nodes, batch)
        if root_nodes.size(0) != change_dense.size(0):
            raise ValueError(
                "root_nodes must contain one root representation per graph"
            )

        change_key = F.normalize(
            self.change_key(change_dense),
            p=2,
            dim=-1,
            eps=self.eps,
        )
        slot_key = F.normalize(
            self.slot_queries,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        # Slots compete for each node.  The second normalization then turns
        # every slot into a distribution over valid nodes.
        slot_score = torch.einsum("bnh,kh->bnk", change_key, slot_key)
        slot_score = slot_score / self.proposal_temperature
        node_responsibility = F.softmax(slot_score, dim=-1)
        node_responsibility = node_responsibility * valid_mask.unsqueeze(-1)
        proposal_attention = node_responsibility.transpose(1, 2)
        proposal_attention = proposal_attention / proposal_attention.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        change_value = self.change_value(change_dense)
        prototypes = torch.matmul(proposal_attention, change_value)
        external_queries = self.query_norm(
            self.tree_query_base.unsqueeze(0)
            + self.prototype_query(prototypes)
            + self.root_query(root_nodes).unsqueeze(1)
        )
        proposal_bias = self.proposal_bias_scale * torch.log(
            proposal_attention.clamp_min(self.eps)
        )
        proposal_bias = proposal_bias.masked_fill(
            ~valid_mask.unsqueeze(1),
            0.0,
        )
        return {
            "external_queries": external_queries,
            "proposal_bias": proposal_bias,
            "proposal_attention": proposal_attention,
            "node_responsibility": node_responsibility,
            "prototypes": prototypes,
            "valid_mask": valid_mask,
        }

    def _off_diagonal_square_mean(self, representation):
        if self.num_slots <= 1:
            return representation.new_zeros(())
        normalized = F.normalize(
            representation,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        gram = torch.matmul(normalized, normalized.transpose(1, 2))
        identity = torch.eye(
            self.num_slots,
            dtype=torch.bool,
            device=representation.device,
        ).unsqueeze(0)
        return gram.masked_select(~identity).pow(2).mean()

    def _classification_margin(self, logits, target):
        target = target.view(-1).long()
        true_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        other_mask = F.one_hot(
            target,
            num_classes=self.num_classes,
        ).bool()
        other_logit = logits.masked_fill(other_mask, -1e9).max(dim=1).values
        return true_logit - other_logit

    def verify(
        self,
        proposal_outputs,
        verified_context,
        tree_attention,
        change_graph,
        semantic_tree_graph,
        target=None,
    ):
        if verified_context is None or tree_attention is None:
            raise ValueError(
                "REPV verification requires Semantic-tree slot contexts "
                "and attention probabilities"
            )
        prototypes = proposal_outputs["prototypes"]
        proposal_attention = proposal_outputs["proposal_attention"]
        if verified_context.shape != prototypes.shape:
            raise ValueError(
                "Semantic-tree verified contexts must match REPV slot shape"
            )
        if tree_attention.shape != proposal_attention.shape:
            raise ValueError(
                "Semantic-tree attention must match REPV proposal attention"
            )

        pair_feature = torch.cat(
            (
                prototypes,
                verified_context,
                prototypes * verified_context,
                (prototypes - verified_context).abs(),
            ),
            dim=-1,
        )
        slot_features = self.slot_fusion(pair_feature)
        importance_logits = self.slot_importance(prototypes).squeeze(-1)
        slot_weights = F.softmax(importance_logits, dim=-1)
        collaboration_raw = (
            slot_weights.unsqueeze(-1) * slot_features
        ).sum(dim=1)
        collaboration_graph = self.collaboration_output(collaboration_raw)

        change_gate = torch.sigmoid(self.change_gate_logit)
        tree_gate = torch.sigmoid(self.tree_gate_logit)
        refined_change_graph = (
            change_graph
            + change_gate * self.change_residual(collaboration_graph)
        )
        refined_tree_graph = (
            semantic_tree_graph
            + tree_gate * self.tree_residual(collaboration_graph)
        )

        zero = collaboration_graph.new_zeros(())
        classification_loss = zero
        feedback_loss = zero
        overlap_loss = zero
        counterfactual_necessity = None
        feedback_target = None
        collaboration_logits = self.collaboration_classifier(
            collaboration_graph
        )
        valid_target = (
            target is not None
            and target.numel() == collaboration_graph.size(0)
        )
        if valid_target:
            target = target.view(-1).long()
            classification_loss = F.cross_entropy(
                collaboration_logits,
                target,
            )

        feedback_scale = self.feedback_scale()
        if self.training and valid_target and feedback_scale > 0.0:
            full_margin = self._classification_margin(
                collaboration_logits,
                target,
            )
            weighted_slots = slot_weights.unsqueeze(-1) * slot_features
            counterfactual_raw = (
                collaboration_raw.unsqueeze(1) - weighted_slots
            )
            counterfactual_graph = self.collaboration_output(
                counterfactual_raw.reshape(-1, self.hidden_dim)
            ).view(
                collaboration_graph.size(0),
                self.num_slots,
                self.hidden_dim,
            )
            counterfactual_logits = self.collaboration_classifier(
                counterfactual_graph
            )
            flat_target = target.unsqueeze(1).expand(
                -1,
                self.num_slots,
            ).reshape(-1)
            counterfactual_margin = self._classification_margin(
                counterfactual_logits.reshape(-1, self.num_classes),
                flat_target,
            ).view(-1, self.num_slots)
            counterfactual_necessity = F.relu(
                full_margin.unsqueeze(1) - counterfactual_margin
            )
            feedback_target = F.softmax(
                counterfactual_necessity
                / self.counterfactual_temperature,
                dim=-1,
            ).detach()
            feedback_loss = F.kl_div(
                F.log_softmax(
                    importance_logits / self.counterfactual_temperature,
                    dim=-1,
                ),
                feedback_target,
                reduction="batchmean",
            )
            overlap = (
                proposal_attention * tree_attention
            ).sum(dim=-1).clamp_min(self.eps)
            overlap_loss = -(
                feedback_target * torch.log(overlap)
            ).sum(dim=-1).mean()

        diversity_loss = 0.5 * (
            self._off_diagonal_square_mean(prototypes)
            + self._off_diagonal_square_mean(tree_attention)
        )
        aux_loss = (
            self.lambda_classification * classification_loss
            + feedback_scale
            * (
                self.lambda_feedback * feedback_loss
                + self.lambda_overlap * overlap_loss
            )
            + self.lambda_diversity * diversity_loss
        )
        return {
            **proposal_outputs,
            "refined_change_graph": refined_change_graph,
            "refined_tree_graph": refined_tree_graph,
            "verified_context": verified_context,
            "tree_attention": tree_attention,
            "slot_features": slot_features,
            "slot_weights": slot_weights,
            "collaboration_graph": collaboration_graph,
            "collaboration_logits": collaboration_logits,
            "counterfactual_necessity": counterfactual_necessity,
            "feedback_target": feedback_target,
            "feedback_scale": collaboration_graph.new_tensor(
                feedback_scale
            ),
            "classification_loss": classification_loss,
            "feedback_loss": feedback_loss,
            "overlap_loss": overlap_loss,
            "diversity_loss": diversity_loss,
            "aux_loss": aux_loss,
        }

