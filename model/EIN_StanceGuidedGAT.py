"""Dual-branch rumor detector with relation-teacher-guided GAT attention.

The propagation branch is selectable between the matched BiGCN and ResGCN
encoders used elsewhere in this repository.  The second branch predicts an
LLM-supervised support/oppose probability for each reply-to-parent edge.  The
predicted support probabilities serve two purposes:

1. they bias the attention used for GAT message passing; and
2. after neighbourhood normalisation, they form a detached soft teacher for
   the *unbiased* GAT attention through a KL-divergence objective.

Raw LLM stance labels are only used by auxiliary losses.  Model predictions
therefore do not require stance labels at inference time.
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


class StanceGuidedGAT(nn.Module):
    """BiGCN/ResGCN plus an LLM-stance relation-teacher GAT branch."""

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
        (
            relation_logits,
            reply_labels,
            gat_edge_index,
            support_probability,
            num_reply_edges,
        ) = self._relation_teacher(data, relation_hidden)

        gat_hidden = self.gat_input_encoder(raw_nodes)
        raw_attention = None
        biased_attention = None
        for layer in self.gat_layers:
            gat_hidden, raw_attention, biased_attention = layer(
                gat_hidden,
                gat_edge_index,
                support_probability,
            )
        gat_graph = self.global_pool(gat_hidden, data.batch)

        fused = self.fusion(torch.cat([backbone_graph, gat_graph], dim=-1))
        output = F.log_softmax(self.classifier(fused), dim=-1)

        relation_loss = self._relation_loss(relation_logits, reply_labels)
        attention_kl = self._attention_kl(
            raw_attention,
            support_probability,
            gat_edge_index,
            reply_labels,
            num_reply_edges,
            raw_nodes.size(0),
        )
        kl_weight = self.lambda_attention_kl * self._kl_schedule()
        self._last_aux_loss = (
            self.lambda_relation * relation_loss
            + kl_weight * attention_kl
        )
        self._last_relation_loss = relation_loss.detach()
        self._last_attention_kl = attention_kl.detach()
        self._last_kl_weight = kl_weight
        self._last_relation_probabilities = (
            F.softmax(relation_logits, dim=-1).detach()
            if relation_logits.numel() > 0
            else relation_logits.detach()
        )
        self._last_raw_attention = raw_attention.detach()
        self._last_biased_attention = biased_attention.detach()
        return output, None, None, None

    def __repr__(self):
        return "{}(backbone={!r})".format(
            self.__class__.__name__,
            self.backbone_name,
        )
