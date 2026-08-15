"""Dual-branch rumor detector with relation-teacher-guided GAT attention.

The propagation branch is selectable between the matched BiGCN and ResGCN
encoders used elsewhere in this repository.  The second branch predicts an
LLM-supervised support/oppose probability for each reply-to-parent edge.  Its
dual-channel mode learns two attention distributions over a shared value space,
then uses the two soft relation probabilities as absolute message gates.
Support has a competitive self-loop; Deny aggregates only replies and therefore
has an exact zero representation when a node has no children.

Across siblings, the non-discretized probabilities form detached relative
teachers for channel-specific raw attention through a KL-divergence objective.
Raw LLM stance labels supervise only the edge classifier and are not consumed
by the prediction path at inference time.  The former single-support-channel
layer is retained behind ``stance_gat_dual_channel: false`` for ablations.

"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter, softmax

from model.DualBackboneOnly import (
    BiGCN_BackboneOnly,
    ResGCN_BackboneOnly,
)


class _EncoderOnlyMixin:
    """Remove the unused classification head from backbone-only models."""

    def _build_classifier(self, args):
        self.fusion = nn.Identity()
        self.classifier = nn.Identity()

    def forward(self, data):
        return self._encode_nodes(data)


class _BiGCNEncoder(_EncoderOnlyMixin, BiGCN_BackboneOnly):
    pass


class _ResGCNEncoder(_EncoderOnlyMixin, ResGCN_BackboneOnly):
    pass


class RelationTeacherGATLayer(nn.Module):
    """GAT layer exposing unbiased and relation-biased attention weights."""

    def __init__(
        self,
        hidden_dim,
        heads=4,
        dropout=0.0,
        negative_slope=0.2,
        relation_bias=1.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        if heads < 1:
            raise ValueError("gat_heads must be at least 1")
        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim ({}) must be divisible by gat_heads ({})".format(
                    hidden_dim,
                    heads,
                )
            )

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)
        self.relation_bias = max(0.0, float(relation_bias))

        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_source = nn.Parameter(
            torch.empty(1, heads, self.head_dim)
        )
        self.attention_target = nn.Parameter(
            torch.empty(1, heads, self.head_dim)
        )
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.bias)
        self.norm.reset_parameters()

    def forward(self, hidden, edge_index, support_probability):
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if support_probability.numel() != edge_index.size(1):
            raise ValueError(
                "support_probability must contain one value per edge"
            )

        num_nodes = hidden.size(0)
        source, target = edge_index
        projected = self.linear(hidden).view(
            num_nodes,
            self.heads,
            self.head_dim,
        )
        source_score = (projected * self.attention_source).sum(dim=-1)
        target_score = (projected * self.attention_target).sum(dim=-1)
        logits = F.leaky_relu(
            source_score[source] + target_score[target],
            negative_slope=self.negative_slope,
        )

        raw_attention = softmax(logits, target, num_nodes=num_nodes)
        relation_log_bias = torch.log(
            support_probability.detach().clamp_min(1e-8)
        )
        biased_logits = (
            logits
            + self.relation_bias * relation_log_bias.unsqueeze(-1)
        )
        biased_attention = softmax(
            biased_logits,
            target,
            num_nodes=num_nodes,
        )
        message_attention = F.dropout(
            biased_attention,
            p=self.dropout,
            training=self.training,
        )
        messages = projected[source] * message_attention.unsqueeze(-1)
        aggregated = scatter(
            messages,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).reshape(num_nodes, self.hidden_dim)
        update = self.output(aggregated) + self.bias
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        output = self.norm(hidden + update)
        return output, raw_attention, biased_attention


class DualChannelRelationTeacherGATLayer(nn.Module):
    """Dual attention over a shared child-value space.

    Both channels start from the same complete node representation and use the
    same value/output transformations.  Across deeper layers their states stay
    parallel and can differ only because of their attention histories and
    per-edge relation gates; no cross-channel state is fed into the next layer.
    Support includes a real self-loop in the neighbourhood softmax; Deny
    contains neither a self-loop nor a synthetic null message.
    """

    def __init__(
        self,
        hidden_dim,
        heads=4,
        dropout=0.0,
        negative_slope=0.2,
        relation_gate_power=1.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        if heads < 1 or hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by a positive gat_heads"
            )
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)
        self.relation_gate_power = max(0.0, float(relation_gate_power))

        # Channel-specific projections are used only for attention scoring.
        # Message values and their output transform are deliberately shared so
        # that Support/Deny semantics differ through weights rather than two
        # unrelated value spaces.
        self.support_attention_linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.deny_attention_linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.value_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.support_attention_source = nn.Parameter(
            torch.empty(1, heads, self.head_dim)
        )
        self.support_attention_target = nn.Parameter(
            torch.empty(1, heads, self.head_dim)
        )
        self.deny_attention_source = nn.Parameter(
            torch.empty(1, heads, self.head_dim)
        )
        self.deny_attention_target = nn.Parameter(
            torch.empty(1, heads, self.head_dim)
        )
        self.output_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.support_attention_linear.weight)
        nn.init.xavier_uniform_(self.deny_attention_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        nn.init.xavier_uniform_(self.support_attention_source)
        nn.init.xavier_uniform_(self.support_attention_target)
        nn.init.xavier_uniform_(self.deny_attention_source)
        nn.init.xavier_uniform_(self.deny_attention_target)
        nn.init.xavier_uniform_(self.output_linear.weight)

    def _relation_gate(self, probability):
        if self.relation_gate_power == 0.0:
            return torch.ones_like(probability)
        return probability.pow(self.relation_gate_power)

    def _attention_logits(self, projected, edge_index, source_att, target_att):
        source, target = edge_index
        source_score = (projected * source_att).sum(dim=-1)
        target_score = (projected * target_att).sum(dim=-1)
        return F.leaky_relu(
            source_score[source] + target_score[target],
            negative_slope=self.negative_slope,
        )

    def forward(
        self,
        support_hidden,
        reply_edge_index,
        relation_probability,
        self_support_prior=1.0,
        deny_hidden=None,
    ):
        if deny_hidden is None:
            deny_hidden = support_hidden
        if deny_hidden.shape != support_hidden.shape:
            raise ValueError(
                "support_hidden and deny_hidden must have the same shape"
            )
        if relation_probability.shape != (reply_edge_index.size(1), 2):
            raise ValueError(
                "relation_probability must have shape [num_reply_edges, 2]"
            )
        num_nodes = support_hidden.size(0)
        reply_source, reply_target = reply_edge_index
        relation_probability = relation_probability.detach().clamp(1e-8, 1.0)
        support_probability = relation_probability[:, 0]
        deny_probability = relation_probability[:, 1]
        self_nodes = torch.arange(num_nodes, device=support_hidden.device)
        self_edges = torch.stack((self_nodes, self_nodes), dim=0)

        support_attention_hidden = self.support_attention_linear(
            support_hidden
        ).view(
            num_nodes, self.heads, self.head_dim
        )
        support_values = self.value_linear(support_hidden).view(
            num_nodes, self.heads, self.head_dim
        )
        support_edges = torch.cat((reply_edge_index, self_edges), dim=1)
        support_raw_logits = self._attention_logits(
            support_attention_hidden,
            support_edges,
            self.support_attention_source,
            self.support_attention_target,
        )
        self_probability = support_hidden.new_full(
            (num_nodes,), float(self_support_prior)
        ).clamp_min(1e-8)
        support_candidate_probability = torch.cat(
            (support_probability, self_probability), dim=0
        )
        support_target = support_edges[1]
        support_raw_attention = softmax(
            support_raw_logits, support_target, num_nodes=num_nodes
        )
        support_relation_gate = self._relation_gate(
            support_candidate_probability
        )
        # The probability is an absolute gate.  Do not renormalise after this
        # multiplication, otherwise uniformly weak Support evidence would be
        # amplified back to a unit-mass neighbourhood distribution.
        support_biased_attention = (
            support_raw_attention * support_relation_gate.unsqueeze(-1)
        )
        support_message_attention = F.dropout(
            support_biased_attention,
            p=self.dropout,
            training=self.training,
        )
        support_messages = (
            support_values[support_edges[0]]
            * support_message_attention.unsqueeze(-1)
        )
        support_aggregated = scatter(
            support_messages,
            support_target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).reshape(num_nodes, self.hidden_dim)
        support_nodes = F.dropout(
            F.elu(self.output_linear(support_aggregated)),
            p=self.dropout,
            training=self.training,
        )

        deny_attention_hidden = self.deny_attention_linear(deny_hidden).view(
            num_nodes, self.heads, self.head_dim
        )
        deny_values = self.value_linear(deny_hidden).view(
            num_nodes, self.heads, self.head_dim
        )
        deny_reply_logits = self._attention_logits(
            deny_attention_hidden,
            reply_edge_index,
            self.deny_attention_source,
            self.deny_attention_target,
        )
        deny_prior = deny_probability
        deny_target = reply_target
        deny_raw_logits = deny_reply_logits
        deny_raw_attention = softmax(
            deny_raw_logits, deny_target, num_nodes=num_nodes
        )
        # Deny has no self candidate.  Its GAT attention determines the
        # relative importance among all children, while the edge classifier's
        # deny probability is an *absolute* message gate.  Do not renormalise
        # after applying this gate: otherwise uniformly small deny
        # probabilities would cancel inside a neighbourhood softmax and still
        # produce a full-strength deny representation.
        deny_relation_gate = self._relation_gate(deny_prior)
        deny_biased_attention = (
            deny_raw_attention * deny_relation_gate.unsqueeze(-1)
        )
        deny_message_attention = F.dropout(
            deny_biased_attention,
            p=self.dropout,
            training=self.training,
        )
        deny_messages = (
            deny_values[reply_source]
            * deny_message_attention.unsqueeze(-1)
        )
        deny_aggregated = scatter(
            deny_messages,
            reply_target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).reshape(num_nodes, self.hidden_dim)
        deny_nodes = F.dropout(
            F.elu(self.output_linear(deny_aggregated)),
            p=self.dropout,
            training=self.training,
        )

        return {
            "support_nodes": support_nodes,
            "deny_nodes": deny_nodes,
            "support_raw_attention": support_raw_attention,
            "support_biased_attention": support_biased_attention,
            "support_relation_gate": support_relation_gate,
            "support_target": support_target,
            "support_prior": support_probability,
            "support_reply_target": reply_target,
            "deny_raw_attention": deny_raw_attention,
            "deny_biased_attention": deny_biased_attention,
            "deny_relation_gate": deny_relation_gate,
            "deny_target": deny_target,
            "deny_prior": deny_prior,
        }


class StanceGuidedGAT(nn.Module):
    """BiGCN/ResGCN plus a stance-routed dual-channel GAT branch."""

    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        args,
        device=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.in_feats = int(in_feats)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.max_hop = max(1, int(getattr(args, "max_hop", 1)))
        self.dropout = float(getattr(args, "dropout", 0.0))
        self.use_dual_channel = bool(
            getattr(args, "stance_gat_dual_channel", False)
        )

        backbone_name = str(
            getattr(args, "stance_gat_backbone", "bigcn")
        ).strip().lower()
        if backbone_name == "bigcn":
            backbone_class = _BiGCNEncoder
        elif backbone_name == "resgcn":
            backbone_class = _ResGCNEncoder
        else:
            raise ValueError(
                "stance_gat_backbone must be 'bigcn' or 'resgcn', got {!r}".format(
                    backbone_name
                )
            )
        self.backbone_name = backbone_name
        self.backbone = backbone_class(
            self.in_feats,
            self.hidden_dim,
            self.hidden_dim,
            self.num_classes,
            args,
            device,
        )

        pool_name = str(getattr(args, "global_pool", "sum")).lower()
        self.global_pool = (
            global_add_pool if "sum" in pool_name else global_mean_pool
        )

        relation_hidden = max(
            1,
            int(
                getattr(
                    args,
                    "stance_relation_hidden_dim",
                    self.hidden_dim,
                )
            ),
        )
        self.relation_node_encoder = nn.Sequential(
            nn.Linear(self.in_feats, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        relation_dropout = float(
            getattr(args, "stance_relation_dropout", 0.0)
        )
        self.edge_relation_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, relation_hidden),
            nn.ReLU(),
            nn.Dropout(relation_dropout),
            nn.Linear(relation_hidden, 2),
        )

        self.gat_input_encoder = nn.Sequential(
            nn.Linear(self.in_feats, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        gat_heads = int(getattr(args, "gat_heads", 4))
        gat_layers = max(1, int(getattr(args, "gat_num_layers", 2)))
        gat_dropout = float(
            getattr(args, "gat_attention_dropout", self.dropout)
        )
        gat_negative_slope = float(
            getattr(args, "gat_negative_slope", 0.2)
        )
        relation_bias = float(
            getattr(args, "stance_relation_attention_bias", 1.0)
        )
        relation_gate_power = float(
            getattr(args, "stance_relation_gate_power", 1.0)
        )
        if self.use_dual_channel:
            self.gat_layers = nn.ModuleList(
                [
                    DualChannelRelationTeacherGATLayer(
                        self.hidden_dim,
                        heads=gat_heads,
                        dropout=gat_dropout,
                        negative_slope=gat_negative_slope,
                        relation_gate_power=relation_gate_power,
                    )
                    for _ in range(gat_layers)
                ]
            )
        else:
            self.gat_layers = nn.ModuleList(
                [
                    RelationTeacherGATLayer(
                        self.hidden_dim,
                        heads=gat_heads,
                        dropout=gat_dropout,
                        negative_slope=gat_negative_slope,
                        relation_bias=relation_bias,
                    )
                    for _ in range(gat_layers)
                ]
            )
        self.dual_graph_fusion = None
        if self.use_dual_channel:
            channel_input_dim = self.hidden_dim * 4
            self.dual_graph_fusion = nn.Sequential(
                nn.Linear(channel_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim),
            )

        fusion_hidden = max(
            self.hidden_dim,
            int(
                getattr(
                    args,
                    "stance_gat_fusion_hidden_dim",
                    self.hidden_dim * 2,
                )
            ),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        relation_weights = getattr(
            args,
            "stance_relation_class_weights",
            None,
        )
        if relation_weights is None:
            self.register_buffer("relation_class_weights", torch.empty(0))
        else:
            weights = torch.tensor(relation_weights, dtype=torch.float32)
            if weights.numel() != 2:
                raise ValueError(
                    "stance_relation_class_weights must contain two values"
                )
            self.register_buffer("relation_class_weights", weights.view(2))

        classification_weights = getattr(
            args,
            "classification_class_weights",
            None,
        )
        if classification_weights is None:
            self.register_buffer(
                "classification_class_weights",
                torch.empty(0),
            )
        else:
            weights = torch.tensor(classification_weights, dtype=torch.float32)
            if weights.numel() != self.num_classes:
                raise ValueError(
                    "classification_class_weights must match num_classes"
                )
            self.register_buffer(
                "classification_class_weights",
                weights.view(self.num_classes),
            )

        self.lambda_relation = max(
            0.0,
            float(getattr(args, "lambda_edge_relation", 0.1)),
        )
        self.lambda_attention_kl = max(
            0.0,
            float(getattr(args, "lambda_attention_kl", 0.05)),
        )
        self.relation_temperature = max(
            1e-3,
            float(getattr(args, "stance_relation_temperature", 1.0)),
        )
        self.self_support_prior = min(
            1.0,
            max(
                1e-4,
                float(getattr(args, "stance_self_support_prior", 1.0)),
            ),
        )
        self.kl_warmup_epochs = max(
            0,
            int(getattr(args, "attention_kl_warmup_epochs", 5)),
        )
        self.kl_ramp_epochs = max(
            0,
            int(getattr(args, "attention_kl_ramp_epochs", 5)),
        )
        self.kl_min_labeled_edges = max(
            1,
            int(
                getattr(
                    args,
                    "attention_kl_min_labeled_edges",
                    1,
                )
            ),
        )

        self.current_epoch = 0
        self._last_aux_loss = None
        self._last_relation_loss = None
        self._last_attention_kl = None
        self._last_kl_weight = 0.0
        self._last_relation_probabilities = None
        self._last_raw_attention = None
        self._last_biased_attention = None
        self._last_support_nodes = None
        self._last_deny_nodes = None
        self._last_support_graph = None
        self._last_deny_graph = None
        self._last_support_attention = None
        self._last_deny_attention = None

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def set_epoch(self, epoch):
        self.current_epoch = max(0, int(epoch))

    def _kl_schedule(self):
        if self.current_epoch < self.kl_warmup_epochs:
            return 0.0
        if self.kl_ramp_epochs == 0:
            return 1.0
        progress = (
            self.current_epoch - self.kl_warmup_epochs + 1
        ) / float(self.kl_ramp_epochs)
        return min(1.0, max(0.0, progress))

    @staticmethod
    def _edge_features(hidden, edge_index):
        source, target = edge_index
        source_hidden = hidden[source]
        target_hidden = hidden[target]
        return torch.cat(
            [
                source_hidden,
                target_hidden,
                source_hidden - target_hidden,
                source_hidden * target_hidden,
            ],
            dim=-1,
        )

    @staticmethod
    def _align_edge_labels(
        source_edge_index,
        source_labels,
        target_edge_index,
        num_nodes,
    ):
        """Align edge attributes by ordered endpoint pair."""
        output = torch.full(
            (target_edge_index.size(1),),
            -1,
            dtype=torch.long,
            device=target_edge_index.device,
        )
        if source_edge_index is None or source_labels is None:
            return output
        source_labels = source_labels.view(-1).long().to(output.device)
        source_edge_index = source_edge_index.to(output.device)
        if source_labels.numel() != source_edge_index.size(1):
            return output
        if target_edge_index.numel() == 0 or source_edge_index.numel() == 0:
            return output

        key_base = max(1, int(num_nodes))
        source_key = (
            source_edge_index[0].long() * key_base
            + source_edge_index[1].long()
        )
        target_key = (
            target_edge_index[0].long() * key_base
            + target_edge_index[1].long()
        )
        sorted_key, order = torch.sort(source_key)
        position = torch.searchsorted(sorted_key, target_key)
        safe_position = position.clamp(max=max(0, sorted_key.numel() - 1))
        valid = position < sorted_key.numel()
        if sorted_key.numel() > 0:
            valid = valid & (sorted_key[safe_position] == target_key)
            output[valid] = source_labels[order[safe_position[valid]]]
        return output

    def _directed_edges_and_stance(self, data):
        directed_edge_index = getattr(data, "directed_edge_index", None)
        if directed_edge_index is None:
            directed_edge_index = data.edge_index
        directed_edge_index = directed_edge_index.long()

        directed_stance = getattr(data, "directed_edge_stance", None)
        if directed_stance is not None:
            directed_stance = directed_stance.view(-1).long()
            if directed_stance.numel() == directed_edge_index.size(1):
                return directed_edge_index, directed_stance

        edge_stance = getattr(data, "edge_stance", None)
        if (
            edge_stance is not None
            and edge_stance.numel() == directed_edge_index.size(1)
        ):
            return directed_edge_index, edge_stance.view(-1).long()
        aligned = self._align_edge_labels(
            getattr(data, "edge_index", None),
            edge_stance,
            directed_edge_index,
            data.x.size(0),
        )
        return directed_edge_index, aligned

    def _reply_to_parent_graph(self, data):
        directed_edge_index, directed_stance = self._directed_edges_and_stance(
            data
        )
        reply_edge_index = torch.stack(
            [directed_edge_index[1], directed_edge_index[0]],
            dim=0,
        )
        return reply_edge_index, directed_stance

    def _relation_loss(self, logits, labels):
        zero = self.classifier.weight.new_zeros(())
        if logits.numel() == 0 or labels is None:
            return zero
        labels = labels.view(-1).long().to(logits.device)
        valid = (labels == 0) | (labels == 1)
        if not valid.any():
            return zero
        weight = (
            self.relation_class_weights
            if self.relation_class_weights.numel() > 0
            else None
        )
        return F.cross_entropy(logits[valid], labels[valid], weight=weight)

    def _attention_kl(
        self,
        raw_attention,
        support_probability,
        edge_index,
        reply_labels,
        num_reply_edges,
        num_nodes,
    ):
        zero = raw_attention.new_zeros(())
        if raw_attention.numel() == 0 or num_reply_edges == 0:
            return zero

        target = edge_index[1]
        teacher_logits = torch.log(
            support_probability.detach().clamp_min(1e-8)
        ) / self.relation_temperature
        teacher = softmax(
            teacher_logits,
            target,
            num_nodes=num_nodes,
        ).detach()
        student = raw_attention.mean(dim=-1).clamp_min(1e-8)
        edge_kl = teacher * (
            torch.log(teacher.clamp_min(1e-8)) - torch.log(student)
        )
        node_kl = scatter(
            edge_kl,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        reply_labels = reply_labels.view(-1).long().to(target.device)
        valid_reply = (reply_labels == 0) | (reply_labels == 1)
        labeled_count = scatter(
            valid_reply.to(dtype=student.dtype),
            target[:num_reply_edges],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        valid_node = labeled_count >= float(self.kl_min_labeled_edges)
        if not valid_node.any():
            return zero
        return node_kl[valid_node].mean()

    def _relation_teacher(self, data, relation_hidden):
        reply_edge_index, reply_labels = self._reply_to_parent_graph(data)
        if reply_edge_index.size(1) == 0:
            relation_logits = relation_hidden.new_zeros((0, 2))
            reply_support = relation_hidden.new_zeros((0,))
        else:
            relation_logits = self.edge_relation_classifier(
                self._edge_features(relation_hidden, reply_edge_index)
            )
            reply_support = F.softmax(relation_logits, dim=-1)[:, 0]

        num_nodes = relation_hidden.size(0)
        self_nodes = torch.arange(num_nodes, device=relation_hidden.device)
        self_edges = torch.stack([self_nodes, self_nodes], dim=0)
        gat_edge_index = torch.cat([reply_edge_index, self_edges], dim=1)
        self_support = relation_hidden.new_full(
            (num_nodes,),
            self.self_support_prior,
        )
        support_probability = torch.cat(
            [reply_support, self_support],
            dim=0,
        ).detach()
        return (
            relation_logits,
            reply_labels,
            gat_edge_index,
            support_probability,
            reply_edge_index.size(1),
        )

    def _dual_relation_teacher(self, data, relation_hidden):
        reply_edge_index, reply_labels = self._reply_to_parent_graph(data)
        if reply_edge_index.size(1) == 0:
            relation_logits = relation_hidden.new_zeros((0, 2))
            relation_probability = relation_hidden.new_zeros((0, 2))
        else:
            relation_logits = self.edge_relation_classifier(
                self._edge_features(relation_hidden, reply_edge_index)
            )
            relation_probability = F.softmax(
                relation_logits / self.relation_temperature,
                dim=-1,
            )
        return (
            relation_logits,
            reply_labels,
            reply_edge_index,
            relation_probability.detach(),
        )

    def _child_attention_kl(
        self,
        child_attention,
        child_probability,
        target,
        reply_labels,
        num_nodes,
    ):
        """Match child-to-child proportions, not total channel mass.

        ``child_probability`` is class-normalised on each edge.  It is
        normalised a second time across siblings to form a relative teacher.
        For Support, ``child_attention`` excludes the self-loop and is
        conditionally renormalised across children before the KL is computed.
        """
        zero = child_attention.new_zeros(())
        if child_attention.numel() == 0:
            return zero
        eps = 1e-8
        child_probability = child_probability.detach().clamp_min(eps)
        teacher_mass = scatter(
            child_probability,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        teacher = child_probability / (
            teacher_mass[target] + eps
        )

        student = child_attention.mean(dim=-1).clamp_min(eps)
        student_mass = scatter(
            student,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        student = student / (student_mass[target] + eps)
        edge_kl = teacher * (
            teacher.clamp_min(eps).log() - student.clamp_min(eps).log()
        )
        node_kl = scatter(
            edge_kl,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        if reply_labels is None:
            return zero
        reply_labels = reply_labels.view(-1).long().to(target.device)
        valid_reply = (reply_labels == 0) | (reply_labels == 1)
        labeled_count = scatter(
            valid_reply.to(dtype=student.dtype),
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        valid_node = labeled_count >= float(self.kl_min_labeled_edges)
        if not valid_node.any():
            return zero
        return node_kl[valid_node].mean()

    def _dual_attention_kl(
        self,
        channel_outputs,
        reply_labels,
        num_reply_edges,
        num_nodes,
    ):
        support_kl = self._child_attention_kl(
            channel_outputs["support_raw_attention"][:num_reply_edges],
            channel_outputs["support_prior"],
            channel_outputs["support_reply_target"],
            reply_labels,
            num_nodes,
        )
        deny_kl = self._child_attention_kl(
            channel_outputs["deny_raw_attention"],
            channel_outputs["deny_prior"],
            channel_outputs["deny_target"],
            reply_labels,
            num_nodes,
        )
        return 0.5 * (support_kl + deny_kl)

    @staticmethod
    def _dual_interaction(support, deny):
        return torch.cat(
            (support, deny, support - deny, support * deny),
            dim=-1,
        )

    def classification_loss(self, output, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.nll_loss(
            output,
            target.view(-1).long(),
            weight=weight,
        )

    def physics_loss(self, U, S, D, true_state):
        return self.classifier.weight.new_zeros(())

    def auxiliary_loss(self):
        if self._last_aux_loss is None:
            return self.classifier.weight.new_zeros(())
        return self._last_aux_loss

    def forward(self, data):
        raw_nodes = data.x.float()
        backbone_nodes = self.backbone(data)
        backbone_graph = self.global_pool(backbone_nodes, data.batch)

        relation_hidden = self.relation_node_encoder(raw_nodes)
        gat_hidden = self.gat_input_encoder(raw_nodes)
        support_nodes = None
        deny_nodes = None
        support_attention = None
        deny_attention = None
        if self.use_dual_channel:
            (
                relation_logits,
                reply_labels,
                reply_edge_index,
                relation_probability,
            ) = self._dual_relation_teacher(data, relation_hidden)
            num_reply_edges = reply_edge_index.size(1)
            support_hidden = gat_hidden
            deny_hidden = gat_hidden
            layer_channel_outputs = []
            for layer in self.gat_layers:
                channel_outputs = layer(
                    support_hidden,
                    reply_edge_index,
                    relation_probability,
                    self_support_prior=self.self_support_prior,
                    deny_hidden=deny_hidden,
                )
                support_hidden = channel_outputs["support_nodes"]
                deny_hidden = channel_outputs["deny_nodes"]
                layer_channel_outputs.append(channel_outputs)
            support_nodes = support_hidden
            deny_nodes = deny_hidden
            support_graph = self.global_pool(support_nodes, data.batch)
            deny_graph = self.global_pool(deny_nodes, data.batch)
            gat_graph = self.dual_graph_fusion(
                self._dual_interaction(support_graph, deny_graph)
            )
            attention_kl = torch.stack(
                [
                    self._dual_attention_kl(
                        outputs,
                        reply_labels,
                        num_reply_edges,
                        raw_nodes.size(0),
                    )
                    for outputs in layer_channel_outputs
                ]
            ).mean()
            # Diagnostics retain the final layer, while the auxiliary KL above
            # supervises child proportions at every propagation depth.
            channel_outputs = layer_channel_outputs[-1]
            raw_attention = channel_outputs["support_raw_attention"]
            biased_attention = channel_outputs["support_biased_attention"]
            support_attention = channel_outputs["support_biased_attention"]
            deny_attention = channel_outputs["deny_biased_attention"]
            cached_relation_probability = relation_probability
        else:
            (
                relation_logits,
                reply_labels,
                gat_edge_index,
                support_probability,
                num_reply_edges,
            ) = self._relation_teacher(data, relation_hidden)
            raw_attention = None
            biased_attention = None
            for layer in self.gat_layers:
                gat_hidden, raw_attention, biased_attention = layer(
                    gat_hidden,
                    gat_edge_index,
                    support_probability,
                )
            gat_graph = self.global_pool(gat_hidden, data.batch)
            attention_kl = self._attention_kl(
                raw_attention,
                support_probability,
                gat_edge_index,
                reply_labels,
                num_reply_edges,
                raw_nodes.size(0),
            )
            cached_relation_probability = (
                F.softmax(relation_logits, dim=-1)
                if relation_logits.numel() > 0
                else relation_logits
            )

        fused = self.fusion(torch.cat([backbone_graph, gat_graph], dim=-1))
        output = F.log_softmax(self.classifier(fused), dim=-1)

        relation_loss = self._relation_loss(relation_logits, reply_labels)
        kl_weight = self.lambda_attention_kl * self._kl_schedule()
        self._last_aux_loss = (
            self.lambda_relation * relation_loss
            + kl_weight * attention_kl
        )
        self._last_relation_loss = relation_loss.detach()
        self._last_attention_kl = attention_kl.detach()
        self._last_kl_weight = kl_weight
        self._last_relation_probabilities = cached_relation_probability.detach()
        self._last_raw_attention = raw_attention.detach()
        self._last_biased_attention = biased_attention.detach()
        self._last_support_nodes = (
            None if support_nodes is None else support_nodes.detach()
        )
        self._last_deny_nodes = (
            None if deny_nodes is None else deny_nodes.detach()
        )
        self._last_support_graph = (
            None
            if support_nodes is None
            else self.global_pool(support_nodes, data.batch).detach()
        )
        self._last_deny_graph = (
            None
            if deny_nodes is None
            else self.global_pool(deny_nodes, data.batch).detach()
        )
        self._last_support_attention = (
            None if support_attention is None else support_attention.detach()
        )
        self._last_deny_attention = (
            None if deny_attention is None else deny_attention.detach()
        )
        return output, None, None, None

    def __repr__(self):
        return "{}(backbone={!r}, dual_channel={!r})".format(
            self.__class__.__name__,
            self.backbone_name,
            self.use_dual_channel,
        )
