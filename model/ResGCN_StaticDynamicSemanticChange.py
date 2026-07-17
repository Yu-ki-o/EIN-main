import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as graph_softmax

from model.ResGCN_UncertaintySemanticChange import (
    ResGCN_UncertaintySemanticChange,
)


class StaticDynamicChangeEncoder(nn.Module):
    """Spatio-temporal change encoder over aligned support/deny trajectories.

    Static change compares support and deny states at the same GNN layer.
    Dynamic change compares how the same node evolves between consecutive
    layers in the two views. The original propagation trajectory supplies a
    shared diagonal-Gaussian relation model; support/deny changes are Values.

    The raw pair distance is symmetric. Spatial normalization is performed
    over incoming graph edges, while temporal normalization is causal.
    """

    def __init__(self, hidden_dim, args):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(getattr(args, "dropout", 0.0))
        self.temperature = max(
            1e-3,
            float(getattr(args, "st_change_attention_temperature", 1.0)),
        )
        self.min_std = max(
            1e-5,
            float(getattr(args, "st_change_min_std", 1e-3)),
        )
        self.max_distance = max(
            1.0,
            float(getattr(args, "st_change_max_wasserstein_distance", 20.0)),
        )
        self.max_depth = max(1, int(getattr(args, "max_hop", 72)))
        self.max_layers = max(
            2,
            int(getattr(args, "n_layers_conv", 2)),
        )
        self.use_ib_sampling = bool(
            getattr(args, "st_change_use_ib_sampling", True)
        )

        self.static_change = self._change_mlp()
        self.dynamic_change = self._change_mlp()

        self.depth_embedding = nn.Embedding(self.max_depth + 2, self.hidden_dim)
        self.layer_embedding = nn.Embedding(self.max_layers + 1, self.hidden_dim)
        self.value_depth_gate = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )
        self.spatial_context = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.temporal_context = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # Shared Q/K projections make the unnormalized Wasserstein relation
        # symmetric. A diagonal Gaussian is sufficient and numerically stable.
        self.spatial_mu = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.spatial_scale = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.temporal_mu = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.temporal_scale = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )

        self.spatial_fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 3,
                self.hidden_dim,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 3,
                self.hidden_dim,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.fusion_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.joint_fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 4,
                self.hidden_dim,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.joint_norm = nn.LayerNorm(self.hidden_dim)

        # The IB split is applied after static/dynamic fusion. Dynamic evidence
        # can therefore remain label-relevant instead of being declared noise.
        self.invariant_mask = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.invariant_mu = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.invariant_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        self.temporal_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self._support_history = None
        self._deny_history = None
        self._original_history = None
        self._depth = None

        self.last_static_nodes = None
        self.last_dynamic_nodes = None
        self.last_invariant_nodes = None
        self.last_variant_nodes = None
        self.last_invariant_mask = None
        self.last_kl_loss = None
        self.last_temporal_prediction_loss = None
        self.last_mask_prior_loss = None
        self.last_spatial_near_attention = None
        self.last_spatial_far_attention = None
        self.last_temporal_near_attention = None
        self.last_temporal_far_attention = None

    def _change_mlp(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )

    def set_trajectories(
        self,
        support_history,
        deny_history,
        original_history,
        depth,
    ):
        if not (
            len(support_history)
            == len(deny_history)
            == len(original_history)
        ):
            raise ValueError("support, deny, and original histories must align")
        if len(support_history) < 2:
            raise ValueError("at least two GNN states are required")
        self._support_history = tuple(support_history)
        self._deny_history = tuple(deny_history)
        self._original_history = tuple(original_history)
        self._depth = depth

    @staticmethod
    def _change_features(first, second):
        delta = second - first
        return torch.cat((delta, delta.abs()), dim=-1)

    def _depth_indices(self, depth):
        return depth.long().clamp(-1, self.max_depth) + 1

    def _diagonal_gaussian(self, hidden, mu_layer, scale_layer):
        mean = mu_layer(hidden)
        std = F.softplus(scale_layer(hidden)) + self.min_std
        return mean, std

    def _spatial_attention(self, context, edge_index, values):
        num_nodes = values.size(0)
        self_nodes = torch.arange(num_nodes, device=values.device)
        self_edges = torch.stack((self_nodes, self_nodes), dim=0)
        edges = torch.cat((edge_index.long(), self_edges), dim=1)
        source, target = edges

        mean, std = self._diagonal_gaussian(
            context,
            self.spatial_mu,
            self.spatial_scale,
        )
        distance = (
            (mean[source] - mean[target]).pow(2)
            + (std[source] - std[target]).pow(2)
        ).mean(dim=-1)
        distance = distance.clamp(max=self.max_distance)

        near_attention = graph_softmax(
            -distance / self.temperature,
            target,
            num_nodes=num_nodes,
        )
        far_attention = graph_softmax(
            distance / self.temperature,
            target,
            num_nodes=num_nodes,
        )
        near_attention = F.dropout(
            near_attention,
            p=self.dropout,
            training=self.training,
        )
        far_attention = F.dropout(
            far_attention,
            p=self.dropout,
            training=self.training,
        )

        near = values.new_zeros(values.shape)
        far = values.new_zeros(values.shape)
        near.index_add_(
            0,
            target,
            near_attention.unsqueeze(-1) * values[source],
        )
        far.index_add_(
            0,
            target,
            far_attention.unsqueeze(-1) * values[source],
        )
        encoded = self.spatial_fusion(
            torch.cat((near, far, near - far), dim=-1)
        )
        self.last_spatial_near_attention = near_attention
        self.last_spatial_far_attention = far_attention
        return encoded + values

    def _temporal_attention(self, context, values):
        # context and values: [num_nodes, num_steps, hidden_dim]
        num_steps = values.size(1)
        if num_steps == 1:
            ones = values.new_ones((values.size(0), 1, 1))
            self.last_temporal_near_attention = ones
            self.last_temporal_far_attention = ones
            return values

        mean, std = self._diagonal_gaussian(
            context,
            self.temporal_mu,
            self.temporal_scale,
        )
        distance = (
            (mean.unsqueeze(2) - mean.unsqueeze(1)).pow(2)
            + (std.unsqueeze(2) - std.unsqueeze(1)).pow(2)
        ).mean(dim=-1)
        distance = distance.clamp(max=self.max_distance)

        causal_mask = torch.triu(
            torch.ones(
                num_steps,
                num_steps,
                dtype=torch.bool,
                device=values.device,
            ),
            diagonal=1,
        )
        near_score = (-distance / self.temperature).masked_fill(
            causal_mask.unsqueeze(0),
            float("-inf"),
        )
        far_score = (distance / self.temperature).masked_fill(
            causal_mask.unsqueeze(0),
            float("-inf"),
        )
        near_attention = F.softmax(near_score, dim=-1)
        far_attention = F.softmax(far_score, dim=-1)
        near_attention = F.dropout(
            near_attention,
            p=self.dropout,
            training=self.training,
        )
        far_attention = F.dropout(
            far_attention,
            p=self.dropout,
            training=self.training,
        )
        near = torch.matmul(near_attention, values)
        far = torch.matmul(far_attention, values)
        encoded = self.temporal_fusion(
            torch.cat((near, far, near - far), dim=-1)
        )
        self.last_temporal_near_attention = near_attention
        self.last_temporal_far_attention = far_attention
        return encoded + values

    def _spatial_contexts(self, original_history, depth):
        depth_hidden = self.depth_embedding(self._depth_indices(depth))
        return [
            self.spatial_context(torch.cat((state, depth_hidden), dim=-1))
            for state in original_history
        ]

    def _static_tokens(self, support_history, deny_history):
        return [
            self.static_change(self._change_features(support, deny))
            for support, deny in zip(support_history, deny_history)
        ]

    def _dynamic_tokens(self, support_history, deny_history):
        dynamic = []
        for layer_index in range(1, len(support_history)):
            support_evolution = (
                support_history[layer_index]
                - support_history[layer_index - 1]
            )
            deny_evolution = (
                deny_history[layer_index]
                - deny_history[layer_index - 1]
            )
            dynamic.append(
                self.dynamic_change(
                    self._change_features(
                        support_evolution,
                        deny_evolution,
                    )
                )
            )
        return dynamic

    def _temporal_contexts(self, original_history):
        original = torch.stack(original_history[1:], dim=1)
        num_steps = original.size(1)
        if num_steps > self.max_layers:
            raise ValueError(
                "trajectory has {} transitions but max_layers is {}".format(
                    num_steps,
                    self.max_layers,
                )
            )
        layer_ids = torch.arange(num_steps, device=original.device)
        layer_hidden = self.layer_embedding(layer_ids).unsqueeze(0)
        layer_hidden = layer_hidden.expand(original.size(0), -1, -1)
        return self.temporal_context(
            torch.cat((original, layer_hidden), dim=-1)
        )

    def _information_bottleneck(self, joint):
        invariant_mask = torch.sigmoid(self.invariant_mask(joint))
        invariant_input = invariant_mask * joint
        variant = (1.0 - invariant_mask) * joint
        mean = self.invariant_mu(invariant_input)
        logvar = self.invariant_logvar(invariant_input).clamp(-8.0, 8.0)
        if self.training and self.use_ib_sampling:
            invariant = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            invariant = mean
        invariant = self.output_norm(invariant)
        kl_loss = 0.5 * (
            mean.pow(2) + logvar.exp() - 1.0 - logvar
        ).mean()
        return invariant, mean, variant, invariant_mask, kl_loss

    def forward(
        self,
        support_nodes,
        deny_nodes,
        edge_index=None,
        support_node_weight=None,
        deny_node_weight=None,
        node_keep=None,
        **kwargs,
    ):
        if self._support_history is None:
            raise RuntimeError("set_trajectories must be called before forward")
        if edge_index is None:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=support_nodes.device,
            )

        support_history = self._support_history
        deny_history = self._deny_history
        original_history = self._original_history
        spatial_contexts = self._spatial_contexts(
            original_history,
            self._depth,
        )

        static_tokens = self._static_tokens(support_history, deny_history)
        dynamic_tokens = self._dynamic_tokens(support_history, deny_history)
        depth_hidden = self.depth_embedding(self._depth_indices(self._depth))
        depth_gate = torch.sigmoid(self.value_depth_gate(depth_hidden))
        static_tokens = [token * depth_gate for token in static_tokens]
        dynamic_tokens = [token * depth_gate for token in dynamic_tokens]

        if support_node_weight is not None and deny_node_weight is not None:
            reliability = 0.5 * (
                support_node_weight.view(-1, 1)
                + deny_node_weight.view(-1, 1)
            )
            reliability = reliability.to(dtype=support_nodes.dtype)
            static_tokens = [token * reliability for token in static_tokens]
            dynamic_tokens = [token * reliability for token in dynamic_tokens]

        static_spatial = [
            self._spatial_attention(context, edge_index, token)
            for context, token in zip(spatial_contexts, static_tokens)
        ]
        dynamic_spatial = [
            self._spatial_attention(context, edge_index, token)
            for context, token in zip(spatial_contexts[1:], dynamic_tokens)
        ]

        temporal_values = torch.stack(dynamic_spatial, dim=1)
        temporal_context = self._temporal_contexts(original_history)
        temporal_encoded = self._temporal_attention(
            temporal_context,
            temporal_values,
        )

        static_nodes = static_spatial[-1]
        dynamic_nodes = temporal_encoded[:, -1]
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((static_nodes, dynamic_nodes), dim=-1))
        )
        residual = self.joint_fusion(
            torch.cat(
                (
                    static_nodes,
                    dynamic_nodes,
                    static_nodes - dynamic_nodes,
                    (static_nodes - dynamic_nodes).abs(),
                ),
                dim=-1,
            )
        )
        joint = self.joint_norm(
            gate * static_nodes + (1.0 - gate) * dynamic_nodes + residual
        )
        (
            invariant,
            invariant_mean,
            variant,
            invariant_mask,
            kl_loss,
        ) = self._information_bottleneck(joint)

        predicted_next = self.temporal_predictor(
            torch.stack(static_spatial[:-1], dim=1)
        )
        target_next = torch.stack(static_spatial[1:], dim=1)
        temporal_prediction_loss = F.smooth_l1_loss(
            predicted_next,
            target_next,
        )

        self.last_static_nodes = static_nodes
        self.last_dynamic_nodes = dynamic_nodes
        self.last_invariant_nodes = invariant_mean
        self.last_variant_nodes = variant
        self.last_invariant_mask = invariant_mask
        self.last_kl_loss = kl_loss
        self.last_temporal_prediction_loss = temporal_prediction_loss
        self.last_mask_prior_loss = (
            invariant_mask.mean()
            - float(kwargs.get("mask_prior", 0.5))
        ).pow(2)

        # Clear graph references so accidental reuse without a new trajectory
        # fails loudly instead of mixing two mini-batches.
        self._support_history = None
        self._deny_history = None
        self._original_history = None
        self._depth = None
        return invariant


class ResGCN_StaticDynamicSemanticChange(
    ResGCN_UncertaintySemanticChange
):
    """Isolated ResGCN model with static and dynamic dual-view evolution."""

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
        if "change" not in self.classification_branch_names:
            raise ValueError(
                "ResGCN_StaticDynamicSemanticChange requires a change branch"
            )

        self.semantic_change_encoder = StaticDynamicChangeEncoder(
            hid_feats,
            args,
        )
        self.static_classifier = nn.Linear(hid_feats, num_classes)
        self.dynamic_classifier = nn.Linear(hid_feats, num_classes)
        self.intervention_classifier = nn.Sequential(
            nn.Linear(hid_feats, hid_feats),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hid_feats, num_classes),
        )

        self.lambda_static_cls = max(
            0.0,
            float(getattr(args, "lambda_static_change_cls", 0.15)),
        )
        self.lambda_dynamic_cls = max(
            0.0,
            float(getattr(args, "lambda_dynamic_change_cls", 0.15)),
        )
        self.lambda_temporal_prediction = max(
            0.0,
            float(getattr(args, "lambda_temporal_prediction", 0.05)),
        )
        self.lambda_information_bottleneck = max(
            0.0,
            float(getattr(args, "lambda_information_bottleneck", 1e-3)),
        )
        self.lambda_variant_intervention = max(
            0.0,
            float(getattr(args, "lambda_variant_intervention", 0.05)),
        )
        self.lambda_static_dynamic_decorr = max(
            0.0,
            float(getattr(args, "lambda_static_dynamic_decorr", 0.01)),
        )
        self.lambda_invariant_mask_prior = max(
            0.0,
            float(getattr(args, "lambda_invariant_mask_prior", 0.01)),
        )
        self.invariant_mask_prior = min(
            1.0,
            max(0.0, float(getattr(args, "invariant_mask_prior", 0.5))),
        )
        self.spatiotemporal_warmup_epochs = max(
            0,
            int(getattr(args, "spatiotemporal_warmup_epochs", 10)),
        )
        self.current_epoch = 0

        self._last_static_change_loss = None
        self._last_dynamic_change_loss = None
        self._last_temporal_prediction_loss = None
        self._last_information_bottleneck_loss = None
        self._last_variant_intervention_loss = None
        self._last_static_dynamic_decorr_loss = None
        self._last_invariant_mask_prior_loss = None
        self._last_static_change_graph = None
        self._last_dynamic_change_graph = None
        self._last_invariant_mask = None

    @property
    def static_dynamic_encoder(self):
        return self.semantic_change_encoder

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.current_epoch = max(0, int(epoch))

    def _encode_view_trajectory(
        self,
        normalized_input,
        edge_index,
        edge_weight,
    ):
        hidden = F.relu(
            self.conv_feat(
                normalized_input,
                edge_index,
                edge_weight=edge_weight,
            )
        )
        history = [hidden]
        for batch_norm, conv in zip(self.bns_conv, self.convs):
            update = F.relu(
                conv(
                    batch_norm(hidden),
                    edge_index,
                    edge_weight=edge_weight,
                )
            )
            update = F.dropout(
                update,
                p=self.dropout,
                training=self.training,
            )
            hidden = hidden + update
            history.append(hidden)
        return history

    def _encode_semantic_views(
        self,
        data,
        node_hidden,
        edge_index,
        support_weight,
        deny_weight,
    ):
        # BatchNorm is evaluated once even though the shared GNN is applied to
        # three edge-weight views.
        normalized_input = self.bn_feat(data.x.float())
        support_history = self._encode_view_trajectory(
            normalized_input,
            edge_index,
            support_weight,
        )
        deny_history = self._encode_view_trajectory(
            normalized_input,
            edge_index,
            deny_weight,
        )
        original_weight = support_weight.new_ones(support_weight.shape)
        original_history = self._encode_view_trajectory(
            normalized_input,
            edge_index,
            original_weight,
        )
        self.static_dynamic_encoder.set_trajectories(
            support_history,
            deny_history,
            original_history,
            self._node_depths(data, edge_index),
        )
        return support_history[-1], deny_history[-1]

    def _pool_auxiliary_nodes(self, nodes, data):
        return self.global_pool(nodes, data.batch)

    def _weighted_cross_entropy(self, logits, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.cross_entropy(logits, target, weight=weight)

    def _decorrelation_loss(self, first, second):
        eps = 1e-6
        if first.size(0) > 1:
            first = first - first.mean(dim=0, keepdim=True)
            second = second - second.mean(dim=0, keepdim=True)
            first = first / first.pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()
            second = second / second.pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()
            cross_covariance = first.t().matmul(second) / first.size(0)
            return cross_covariance.pow(2).mean()
        first = F.normalize(first, dim=-1, eps=eps)
        second = F.normalize(second, dim=-1, eps=eps)
        return (first * second).sum(dim=-1).pow(2).mean()

    def _warmup_factor(self):
        if self.spatiotemporal_warmup_epochs <= 0:
            return 1.0
        return min(
            1.0,
            float(self.current_epoch + 1)
            / float(self.spatiotemporal_warmup_epochs),
        )

    def _intervention_loss(self, invariant_graph, variant_graph, target):
        if invariant_graph.size(0) <= 1:
            return invariant_graph.new_zeros(())
        shift = int(
            torch.randint(
                1,
                invariant_graph.size(0),
                (1,),
                device=invariant_graph.device,
            ).item()
        )
        permuted_variant = torch.roll(variant_graph, shifts=shift, dims=0)
        original_logits = self.intervention_classifier(
            invariant_graph + variant_graph
        )
        intervened_logits = self.intervention_classifier(
            invariant_graph + permuted_variant
        )
        label_loss = 0.5 * (
            self._weighted_cross_entropy(original_logits, target)
            + self._weighted_cross_entropy(intervened_logits, target)
        )
        consistency = F.kl_div(
            F.log_softmax(intervened_logits, dim=-1),
            F.softmax(original_logits.detach(), dim=-1),
            reduction="batchmean",
        )
        return label_loss + consistency

    def forward(self, data):
        output = super().forward(data)
        encoder = self.static_dynamic_encoder
        target = data.y.view(-1).long()

        static_graph = self._pool_auxiliary_nodes(
            encoder.last_static_nodes,
            data,
        )
        dynamic_graph = self._pool_auxiliary_nodes(
            encoder.last_dynamic_nodes,
            data,
        )
        invariant_graph = self._pool_auxiliary_nodes(
            encoder.last_invariant_nodes,
            data,
        )
        variant_graph = self._pool_auxiliary_nodes(
            encoder.last_variant_nodes,
            data,
        )

        static_loss = self._weighted_cross_entropy(
            self.static_classifier(static_graph),
            target,
        )
        dynamic_loss = self._weighted_cross_entropy(
            self.dynamic_classifier(dynamic_graph),
            target,
        )
        decorrelation_loss = self._decorrelation_loss(
            static_graph,
            dynamic_graph,
        )
        intervention_loss = self._intervention_loss(
            invariant_graph,
            variant_graph,
            target,
        )
        warmup = self._warmup_factor()
        mask_prior_loss = (
            encoder.last_invariant_mask.mean() - self.invariant_mask_prior
        ).pow(2)

        static_term = self.lambda_static_cls * static_loss
        dynamic_term = self.lambda_dynamic_cls * dynamic_loss
        temporal_term = (
            self.lambda_temporal_prediction
            * encoder.last_temporal_prediction_loss
        )
        ib_term = (
            warmup
            * self.lambda_information_bottleneck
            * encoder.last_kl_loss
        )
        intervention_term = (
            warmup
            * self.lambda_variant_intervention
            * intervention_loss
        )
        decorrelation_term = (
            warmup
            * self.lambda_static_dynamic_decorr
            * decorrelation_loss
        )
        mask_prior_term = (
            warmup
            * self.lambda_invariant_mask_prior
            * mask_prior_loss
        )
        self._last_aux_loss = self._last_aux_loss + (
            static_term
            + dynamic_term
            + temporal_term
            + ib_term
            + intervention_term
            + decorrelation_term
            + mask_prior_term
        )

        self._last_static_change_loss = static_term.detach()
        self._last_dynamic_change_loss = dynamic_term.detach()
        self._last_temporal_prediction_loss = temporal_term.detach()
        self._last_information_bottleneck_loss = ib_term.detach()
        self._last_variant_intervention_loss = intervention_term.detach()
        self._last_static_dynamic_decorr_loss = decorrelation_term.detach()
        self._last_invariant_mask_prior_loss = mask_prior_term.detach()
        self._last_static_change_graph = static_graph.detach()
        self._last_dynamic_change_graph = dynamic_graph.detach()
        self._last_invariant_mask = encoder.last_invariant_mask.detach()
        return output

    def __repr__(self):
        return self.__class__.__name__
