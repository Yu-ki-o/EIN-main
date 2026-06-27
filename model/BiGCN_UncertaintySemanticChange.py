import math
import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from model.semantic_change_encoder import build_semantic_change_encoder


class EdgeRelationUncertaintyRouter(nn.Module):
    """
    Predicts support/deny edge relations and samples edge reliability.

    Relation uncertainty is the normalized entropy of the two-class
    distribution. A differentiable Binary Concrete sample decides how much of
    the edge remains reliable before its mass is divided between the support
    and deny semantic views.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        relation_hidden = int(
            getattr(args, "relation_hidden_dim", hidden_dim)
        )
        self.relation_temperature = max(
            1e-6,
            float(getattr(args, "relation_temperature", 1.0)),
        )
        self.sample_temperature = max(
            1e-6,
            float(getattr(args, "uncertainty_sample_temperature", 0.5)),
        )
        self.route_temperature = max(
            1e-6,
            float(getattr(args, "stance_route_temperature", 0.5)),
        )
        self.hard_stance_route = bool(
            getattr(args, "stance_route_hard", True)
        )
        self.keep_floor = min(
            max(
                float(getattr(args, "uncertainty_keep_floor", 0.05)),
                0.0,
            ),
            1.0,
        )
        self.use_degree_importance = bool(
            getattr(args, "use_degree_importance", True)
        )
        self.degree_importance_strength = max(
            0.0,
            float(getattr(args, "degree_importance_strength", 1.0)),
        )
        self.warmup_epochs = max(
            0,
            int(getattr(args, "uncertainty_sampling_warmup_epochs", 5)),
        )
        self.register_buffer(
            "_current_epoch",
            torch.zeros((), dtype=torch.long),
        )
        self.eps = 1e-6

        self.relation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, relation_hidden),
            nn.ReLU(),
            nn.Linear(relation_hidden, relation_hidden),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(relation_hidden, 2)

    def relation_probabilities(self, logits):
        return F.softmax(logits / self.relation_temperature, dim=-1)

    def normalized_entropy(self, probabilities):
        probabilities = probabilities.clamp_min(self.eps)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        return (entropy / math.log(2.0)).clamp(0.0, 1.0)

    def reliability_probability(
        self,
        uncertainty,
        child_degree_importance=None,
    ):
        expected_keep = (1.0 - uncertainty).clamp(0.0, 1.0)
        if (
            self.use_degree_importance
            and child_degree_importance is not None
        ):
            child_degree_importance = child_degree_importance.clamp(
                0.0,
                1.0,
            )
            exponent = 1.0 + self.degree_importance_strength * (
                1.0 - child_degree_importance
            )
            expected_keep = expected_keep.clamp_min(self.eps).pow(exponent)
        return (
            self.keep_floor
            + (1.0 - self.keep_floor) * expected_keep
        )

    def soft_bernoulli_sample(self, keep_probability):
        keep_probability = keep_probability.clamp(
            self.eps,
            1.0 - self.eps,
        )
        if not self.training:
            return keep_probability

        uniform = torch.rand_like(keep_probability).clamp(
            self.eps,
            1.0 - self.eps,
        )
        logistic_noise = uniform.log() - torch.log1p(-uniform)
        keep_logit = torch.logit(keep_probability, eps=self.eps)
        return torch.sigmoid(
            (keep_logit + logistic_noise) / self.sample_temperature
        )

    def set_epoch(self, epoch):
        self._current_epoch.fill_(max(0, int(epoch)))

    @property
    def current_epoch(self):
        return int(self._current_epoch.item())

    def stance_route(self, logits, probabilities):
        if self.current_epoch < self.warmup_epochs:
            return probabilities
        if self.training:
            return F.gumbel_softmax(
                logits,
                tau=self.route_temperature,
                hard=self.hard_stance_route,
                dim=-1,
            )
        if self.hard_stance_route:
            predicted = probabilities.argmax(dim=-1)
            return F.one_hot(
                predicted,
                num_classes=2,
            ).to(dtype=probabilities.dtype)
        return probabilities

    def forward(
        self,
        node_hidden,
        edge_index,
        child_degree_importance=None,
    ):
        if edge_index.numel() == 0:
            empty_logits = node_hidden.new_zeros((0, 2))
            empty_scalar = node_hidden.new_zeros((0,))
            return (
                empty_logits,
                empty_logits,
                empty_scalar,
                empty_scalar,
                empty_scalar,
                empty_scalar,
            )

        src, dst = edge_index
        parent = node_hidden[src]
        child = node_hidden[dst]
        edge_features = torch.cat(
            (
                parent,
                child,
                child - parent,
                parent * child,
            ),
            dim=-1,
        )
        logits = self.logit_head(self.relation_encoder(edge_features))
        probabilities = self.relation_probabilities(logits)
        uncertainty = self.normalized_entropy(probabilities)

        keep_probability = self.reliability_probability(
            uncertainty,
            child_degree_importance,
        )
        if self.current_epoch < self.warmup_epochs:
            keep_sample = torch.ones_like(keep_probability)
        else:
            keep_sample = self.soft_bernoulli_sample(keep_probability)

        # Two-stage routing:
        #   1. keep_sample decides whether the uncertain edge is retained;
        #   2. stance_route decides which semantic view receives it.
        route = self.stance_route(logits, probabilities)
        support_weight = keep_sample * route[:, 0]
        deny_weight = keep_sample * route[:, 1]
        return (
            logits,
            probabilities,
            uncertainty,
            keep_sample,
            support_weight,
            deny_weight,
        )


class BiGCN_UncertaintySemanticChange(nn.Module):
    """
    Uncertainty-routed support/deny dual-view rumor detector.

    Pipeline:
      edge relation logits
      -> entropy uncertainty
      -> Binary Concrete reliability sampling
      -> support and deny weighted propagation graphs
      -> shared bidirectional GCN encoders
      -> node-aligned MLP semantic change encoding
      -> support/unknown/deny depth-trend GRU
      -> configurable graph-level branch fusion for classification.

    The unknown component is uncertainty mass, not a separately supervised
    stance class.
    """

    backbone_type = "bigcn"

    def __init__(
        self,
        in_feats,
        hid_feats,
        out_feats,
        num_classes,
        args,
        device,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_dim = int(hid_feats)
        self.max_hop = int(args.max_hop)
        self.dropout = float(getattr(args, "dropout", 0.1))
        self.use_trend_graph = bool(
            getattr(args, "use_trend_graph", True)
        )
        self.use_node_keep_in_change_pool = bool(
            getattr(args, "use_node_keep_in_change_pool", True)
        )
        requested_fusion_mode = getattr(
            args,
            "classification_fusion_mode",
            None,
        )
        if requested_fusion_mode is None:
            requested_fusion_mode = (
                "original_change_trend"
                if self.use_trend_graph
                else "original_change"
            )
        self.classification_fusion_mode = str(
            requested_fusion_mode
        ).strip().lower()
        fusion_mode_branches = {
            "original_change": ("original", "change"),
            "original_change_trend": (
                "original",
                "change",
                "trend",
            ),
            "support_deny_change": (
                "support",
                "deny",
                "change",
            ),
        }
        if self.classification_fusion_mode not in fusion_mode_branches:
            raise ValueError(
                "classification_fusion_mode must be one of {}, got {}".format(
                    sorted(fusion_mode_branches),
                    self.classification_fusion_mode,
                )
            )
        self.classification_branch_names = fusion_mode_branches[
            self.classification_fusion_mode
        ]
        self.lambda_edge_relation = max(
            0.0,
            float(getattr(args, "lambda_edge_relation_aux", 0.1)),
        )
        self.lambda_edge_relation_warmup = max(
            self.lambda_edge_relation,
            float(
                getattr(
                    args,
                    "lambda_edge_relation_warmup",
                    self.lambda_edge_relation,
                )
            ),
        )
        class_weights = getattr(args, "classification_class_weights", None)
        if class_weights is None:
            self.register_buffer(
                "classification_class_weights",
                torch.empty(0),
            )
        else:
            self.register_buffer(
                "classification_class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )

        pool_name = str(getattr(args, "global_pool", "mean")).lower()
        self.pool_is_sum = "sum" in pool_name
        self.global_pool = (
            global_add_pool if self.pool_is_sum else global_mean_pool
        )

        self.node_projection = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.ReLU(),
        )
        self.root_context = nn.Sequential(
            nn.Linear(hid_feats * 2, hid_feats),
            nn.ReLU(),
        )
        self.edge_router = EdgeRelationUncertaintyRouter(
            hid_feats,
            args=args,
        )

        self._build_view_backbone(
            in_feats,
            hid_feats,
            out_feats,
            args,
        )

        change_name = getattr(args, "semantic_change_encoder", "mlp")
        change_hidden = int(
            getattr(args, "semantic_change_hidden_dim", hid_feats)
        )
        self.semantic_change_encoder = build_semantic_change_encoder(
            change_name,
            input_dim=hid_feats,
            output_dim=hid_feats,
            hidden_dim=change_hidden,
            dropout=self.dropout,
        )

        trend_hidden = int(
            getattr(args, "uncertainty_trend_hidden_dim", hid_feats)
        )
        self.uncertainty_trend_encoder = nn.GRU(
            input_size=5,
            hidden_size=trend_hidden,
            batch_first=True,
        )
        self.trend_projection = (
            nn.Identity()
            if trend_hidden == hid_feats
            else nn.Linear(trend_hidden, hid_feats)
        )

        fusion_hidden = int(
            getattr(args, "classification_fusion_hidden_dim", hid_feats * 2)
        )
        fusion_branch_count = len(self.classification_branch_names)
        self.fusion = nn.Sequential(
            nn.Linear(hid_feats * fusion_branch_count, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_hidden, hid_feats),
            nn.ReLU(),
            nn.LayerNorm(hid_feats),
        )
        self.classifier = nn.Linear(hid_feats, num_classes)

        self._last_aux_loss = None
        self._last_edge_probabilities = None
        self._last_edge_uncertainty = None
        self._last_keep_sample = None
        self._last_support_weight = None
        self._last_deny_weight = None
        self._last_original_graph = None
        self._last_support_graph = None
        self._last_deny_graph = None
        self._last_change_nodes = None
        self._last_change_graph = None
        self._last_trend_sequence = None
        self._last_node_state_sequence = None
        self._last_node_keep = None
        self._last_child_degree_importance = None

    def _build_view_backbone(
        self,
        in_feats,
        hid_feats,
        out_feats,
        args,
    ):
        if self.backbone_type != "bigcn":
            raise ValueError(
                "unsupported backbone_type: {}".format(self.backbone_type)
            )
        self.td_conv1 = GCNConv(in_feats, hid_feats)
        self.td_conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.bu_conv1 = GCNConv(in_feats, hid_feats)
        self.bu_conv2 = GCNConv(hid_feats + in_feats, out_feats)
        branch_dim = out_feats + hid_feats
        self.direction_fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, hid_feats),
            nn.ReLU(),
        )

    def set_epoch(self, epoch):
        self.edge_router.set_epoch(epoch)

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def auxiliary_loss(self):
        if self._last_aux_loss is None:
            return self.classifier.weight.new_zeros(())
        return self._last_aux_loss

    def classification_loss(self, output, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.nll_loss(output, target, weight=weight)

    def physics_loss(self, U, S, D, true_state):
        # This model replaces the fixed EIN compartment recurrence with a
        # learned uncertainty trend encoder. U/S/D are returned for logging and
        # interface compatibility, not supervised by the original physics loss.
        return self.classifier.weight.new_zeros(())

    def _root_indices(self, data):
        if hasattr(data, "ptr"):
            return data.ptr[:-1].to(device=data.batch.device)
        is_root = torch.ones(
            data.batch.size(0),
            dtype=torch.bool,
            device=data.batch.device,
        )
        is_root[1:] = data.batch[1:] != data.batch[:-1]
        return is_root.nonzero(as_tuple=False).view(-1)

    def _add_root_context(self, node_hidden, data):
        roots = self._root_indices(data)
        root_for_node = roots[data.batch.long()]
        return self.root_context(
            torch.cat((node_hidden, node_hidden[root_for_node]), dim=-1)
        )

    def _reverse_edges(self, edge_index):
        return torch.stack((edge_index[1], edge_index[0]), dim=0)

    def _extend_root_features(self, node_features, data):
        roots = self._root_indices(data)
        return node_features[roots][data.batch.long()]

    def _encode_bigcn_direction(
        self,
        data,
        edge_index,
        edge_weight,
        conv1,
        conv2,
    ):
        raw_nodes = data.x.float()
        hidden_first = conv1(
            raw_nodes,
            edge_index,
            edge_weight=edge_weight,
        )
        root_raw = self._extend_root_features(raw_nodes, data)
        hidden = torch.cat((hidden_first, root_raw), dim=-1)
        hidden = F.relu(hidden)
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        hidden = conv2(hidden, edge_index, edge_weight=edge_weight)
        hidden = F.relu(hidden)
        root_hidden = self._extend_root_features(hidden_first, data)
        return torch.cat((hidden, root_hidden), dim=-1)

    def _encode_semantic_view(
        self,
        data,
        node_hidden,
        edge_index,
        edge_weight,
    ):
        top_down = self._encode_bigcn_direction(
            data,
            edge_index,
            edge_weight,
            self.td_conv1,
            self.td_conv2,
        )
        bottom_up = self._encode_bigcn_direction(
            data,
            self._reverse_edges(edge_index),
            edge_weight,
            self.bu_conv1,
            self.bu_conv2,
        )
        return self.direction_fusion(
            torch.cat((top_down, bottom_up), dim=-1)
        )

    def _edge_relation_loss(self, logits, edge_stance):
        zero = self.classifier.weight.new_zeros(())
        if edge_stance is None or logits.numel() == 0:
            return zero
        labels = edge_stance.view(-1).long()
        if labels.numel() != logits.size(0):
            return zero
        valid = (labels == 0) | (labels == 1)
        if not valid.any():
            return zero
        valid_labels = labels[valid]
        class_count = torch.bincount(
            valid_labels,
            minlength=2,
        ).to(dtype=logits.dtype)
        if (class_count > 0).all():
            class_weight = (
                class_count.sum() / (2.0 * class_count)
            )
        else:
            class_weight = None
        relation_weight = (
            self.lambda_edge_relation_warmup
            if self.edge_router.current_epoch < self.edge_router.warmup_epochs
            else self.lambda_edge_relation
        )
        return relation_weight * F.cross_entropy(
            logits[valid],
            valid_labels,
            weight=class_weight,
        )

    def _node_depths(self, data, edge_index):
        num_nodes = data.x.size(0)
        depth = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=data.x.device,
        )
        roots = self._root_indices(data)
        depth[roots] = 0
        if edge_index.numel() == 0:
            return depth

        src, dst = edge_index
        for _ in range(min(self.max_hop + 1, num_nodes)):
            parent_depth = depth[src]
            candidate = parent_depth + 1
            update = (parent_depth >= 0) & (
                (depth[dst] < 0) | (candidate < depth[dst])
            )
            depth[dst[update]] = candidate[update]
        return depth

    def _build_root_connected_keep(self, data, keep_sample):
        num_nodes = data.x.size(0)
        node_keep = keep_sample.new_zeros(num_nodes)
        roots = self._root_indices(data)
        node_keep[roots] = 1.0
        if data.edge_index.numel() == 0:
            return node_keep

        src, dst = data.edge_index
        depth = self._node_depths(data, data.edge_index)
        depth_src = depth[src]
        depth_dst = depth[dst]
        for depth_id in range(1, self.max_hop + 1):
            edge_mask = (
                (depth_dst == depth_id)
                & (depth_src == depth_id - 1)
            )
            edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
            if edge_ids.numel() == 0:
                continue
            parent = src[edge_ids]
            child = dst[edge_ids]
            node_keep[child] = (
                node_keep[parent] * keep_sample[edge_ids]
            )
        return node_keep

    def _child_degree_importance(self, data):
        num_edges = data.edge_index.size(1)
        if num_edges == 0:
            return data.x.new_zeros((0,))

        src, dst = data.edge_index
        out_degree = torch.bincount(
            src,
            minlength=data.x.size(0),
        ).to(dtype=data.x.dtype)

        # Only non-root child nodes are sampling candidates. Excluding roots
        # from the graph-wise maximum prevents a high-degree source post from
        # making every reply appear structurally unimportant.
        roots = self._root_indices(data)
        out_degree = out_degree.clone()
        out_degree[roots] = 0.0
        degree_score = torch.log1p(out_degree)
        graph_max = global_max_pool(
            degree_score.unsqueeze(-1),
            data.batch,
        ).squeeze(-1)
        node_importance = degree_score / graph_max[
            data.batch.long()
        ].clamp_min(1e-6)
        return node_importance[dst].clamp(0.0, 1.0)

    def _pool_root_connected_nodes(
        self,
        node_hidden,
        node_keep,
        batch,
    ):
        weight = node_keep.unsqueeze(-1)
        weighted_sum = global_add_pool(node_hidden * weight, batch)
        if self.pool_is_sum:
            return weighted_sum
        weight_sum = global_add_pool(weight, batch)
        return weighted_sum / weight_sum.clamp_min(1e-6)

    def _build_view_node_weight(self, data, edge_weight):
        node_weight = edge_weight.new_zeros(data.x.size(0))
        roots = self._root_indices(data)
        # The source post is the shared semantic anchor of both views.
        node_weight[roots] = 1.0
        if data.edge_index.numel() == 0:
            return node_weight

        src, dst = data.edge_index
        depth = self._node_depths(data, data.edge_index)
        depth_src = depth[src]
        depth_dst = depth[dst]
        for depth_id in range(1, self.max_hop + 1):
            edge_mask = (
                (depth_dst == depth_id)
                & (depth_src == depth_id - 1)
            )
            edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
            if edge_ids.numel() == 0:
                continue
            node_weight[dst[edge_ids]] = edge_weight[edge_ids]
        return node_weight

    def _encode_original_graph(self, data, node_hidden):
        # The original-graph branch keeps every propagation edge. It shares the
        # dataset-specific BiGCN/ResGCN backbone with the semantic views, but is
        # independent of uncertainty sampling and support/deny routing.
        original_edge_weight = data.x.new_ones(data.edge_index.size(1))
        original_nodes = self._encode_semantic_view(
            data,
            node_hidden,
            data.edge_index,
            original_edge_weight,
        )
        return self.global_pool(original_nodes, data.batch)

    def _build_uncertainty_trend(
        self,
        data,
        probabilities,
        keep_sample,
    ):
        batch_size = int(data.num_hop.view(-1).size(0))
        trend = data.x.new_zeros(batch_size, self.max_hop, 5)
        node_state_sequence = data.x.new_zeros(
            batch_size,
            self.max_hop,
            3,
        )
        if data.edge_index.numel() == 0:
            self._last_node_state_sequence = node_state_sequence
            return trend

        src, dst = data.edge_index
        depth = self._node_depths(data, data.edge_index)
        valid_node = (depth >= 1) & (depth <= self.max_hop)
        if not valid_node.any():
            self._last_node_state_sequence = node_state_sequence
            return trend

        num_nodes = data.x.size(0)
        node_state = data.x.new_zeros(num_nodes, 3)
        roots = self._root_indices(data)
        # State order: support, uncertain, deny.
        node_state[roots, 0] = 1.0

        depth_src = depth[src]
        depth_dst = depth[dst]
        for depth_id in range(1, self.max_hop + 1):
            edge_mask = (
                (depth_dst == depth_id)
                & (depth_src == depth_id - 1)
            )
            edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
            if edge_ids.numel() == 0:
                continue
            parent = src[edge_ids]
            child = dst[edge_ids]
            parent_state = node_state[parent]
            relation = probabilities[edge_ids]
            reliable = keep_sample[edge_ids].unsqueeze(-1)

            parent_support = parent_state[:, 0:1]
            parent_uncertain = parent_state[:, 1:2]
            parent_deny = parent_state[:, 2:3]
            relation_support = relation[:, 0:1]
            relation_deny = relation[:, 1:2]

            child_support = reliable * (
                parent_support * relation_support
                + parent_deny * relation_deny
            )
            child_deny = reliable * (
                parent_support * relation_deny
                + parent_deny * relation_support
            )
            child_uncertain = (
                (1.0 - reliable)
                + reliable * parent_uncertain
            )
            child_state = torch.cat(
                (child_support, child_uncertain, child_deny),
                dim=-1,
            )
            node_state[child] = child_state / child_state.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)

        valid_batch = data.batch[valid_node].long()
        valid_depth = depth[valid_node].long() - 1
        flat_index = valid_batch * self.max_hop + valid_depth
        flat_size = batch_size * self.max_hop

        state_sum = data.x.new_zeros(flat_size, 3)
        node_count = data.x.new_zeros(flat_size, 1)
        state_sum.index_add_(0, flat_index, node_state[valid_node])
        node_count.index_add_(
            0,
            flat_index,
            torch.ones_like(valid_depth, dtype=data.x.dtype).unsqueeze(-1),
        )
        state_mean = state_sum / node_count.clamp_min(1.0)
        node_state_sequence = state_mean.view(
            batch_size,
            self.max_hop,
            3,
        )

        count_sequence = node_count.view(
            batch_size,
            self.max_hop,
            1,
        )
        graph_node_count = torch.bincount(
            data.batch,
            minlength=batch_size,
        ).to(dtype=data.x.dtype).view(-1, 1, 1)
        count_feature = torch.log1p(count_sequence) / torch.log1p(
            graph_node_count.clamp_min(1.0)
        )
        previous_count = torch.cat(
            (
                count_sequence.new_zeros(batch_size, 1, 1),
                count_sequence[:, :-1],
            ),
            dim=1,
        )
        growth_feature = (
            torch.log1p(count_sequence)
            - torch.log1p(previous_count)
        )
        trend = torch.cat(
            (
                node_state_sequence,
                count_feature,
                growth_feature,
            ),
            dim=-1,
        )
        has_node = count_sequence > 0
        trend = torch.where(
            has_node.expand_as(trend),
            trend,
            torch.zeros_like(trend),
        )
        self._last_node_state_sequence = node_state_sequence.detach()
        return trend

    def _encode_trend(self, trend, num_hop):
        hidden, _ = self.uncertainty_trend_encoder(trend)
        last_index = (
            num_hop.view(-1).long().clamp(1, self.max_hop) - 1
        )
        batch_index = torch.arange(
            hidden.size(0),
            device=hidden.device,
        )
        return self.trend_projection(hidden[batch_index, last_index])

    def forward(self, data):
        node_hidden = self.node_projection(data.x.float())
        node_hidden = self._add_root_context(node_hidden, data)
        original_graph = None
        if "original" in self.classification_branch_names:
            original_graph = self._encode_original_graph(data, node_hidden)
        child_degree_importance = self._child_degree_importance(data)

        (
            relation_logits,
            probabilities,
            edge_uncertainty,
            keep_sample,
            support_weight,
            deny_weight,
        ) = self.edge_router(
            node_hidden,
            data.edge_index,
            child_degree_importance,
        )
        node_keep = self._build_root_connected_keep(
            data,
            keep_sample,
        )
        if data.edge_index.numel() > 0:
            parent_keep = node_keep[data.edge_index[0]]
            support_weight = support_weight * parent_keep
            deny_weight = deny_weight * parent_keep

        support_nodes = self._encode_semantic_view(
            data,
            node_hidden,
            data.edge_index,
            support_weight,
        )
        deny_nodes = self._encode_semantic_view(
            data,
            node_hidden,
            data.edge_index,
            deny_weight,
        )
        support_node_weight = self._build_view_node_weight(
            data,
            support_weight,
        )
        deny_node_weight = self._build_view_node_weight(
            data,
            deny_weight,
        )
        support_graph = self._pool_root_connected_nodes(
            support_nodes,
            support_node_weight,
            data.batch,
        )
        deny_graph = self._pool_root_connected_nodes(
            deny_nodes,
            deny_node_weight,
            data.batch,
        )
        change_nodes = self.semantic_change_encoder(
            support_nodes,
            deny_nodes,
        )

        if self.use_node_keep_in_change_pool:
            change_graph = self._pool_root_connected_nodes(
                change_nodes,
                node_keep,
                data.batch,
            )
        else:
            change_graph = self.global_pool(change_nodes, data.batch)

        trend_sequence = self._build_uncertainty_trend(
            data,
            probabilities,
            keep_sample,
        )
        trend_graph = self._encode_trend(
            trend_sequence,
            data.num_hop,
        )

        graph_branches = {
            "original": original_graph,
            "support": support_graph,
            "deny": deny_graph,
            "change": change_graph,
            "trend": trend_graph,
        }
        classification_graphs = [
            graph_branches[name]
            for name in self.classification_branch_names
        ]
        fused = self.fusion(
            torch.cat(classification_graphs, dim=-1)
        )
        logits = self.classifier(fused)

        self._last_aux_loss = self._edge_relation_loss(
            relation_logits,
            getattr(data, "edge_stance", None),
        )
        self._last_edge_probabilities = probabilities.detach()
        self._last_edge_uncertainty = edge_uncertainty.detach()
        self._last_keep_sample = keep_sample.detach()
        self._last_support_weight = support_weight.detach()
        self._last_deny_weight = deny_weight.detach()
        self._last_original_graph = (
            None if original_graph is None else original_graph.detach()
        )
        self._last_support_graph = support_graph.detach()
        self._last_deny_graph = deny_graph.detach()
        self._last_change_nodes = change_nodes.detach()
        self._last_change_graph = change_graph.detach()
        self._last_trend_sequence = trend_sequence.detach()
        self._last_node_keep = node_keep.detach()
        self._last_child_degree_importance = (
            child_degree_importance.detach()
        )

        support_sequence = trend_sequence[:, :, 0:1]
        unknown_sequence = trend_sequence[:, :, 1:2]
        deny_sequence = trend_sequence[:, :, 2:3]
        return (
            F.log_softmax(logits, dim=-1),
            unknown_sequence,
            support_sequence,
            deny_sequence,
        )

    def __repr__(self):
        return self.__class__.__name__
