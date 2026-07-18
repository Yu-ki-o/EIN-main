import math
import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, kl_divergence
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax, to_dense_batch

from model.collective_revision import CollectiveRevisionEncoder
from model.semantic_change_encoder import build_semantic_change_encoder


class EdgeRelationUncertaintyRouter(nn.Module):
    """
    Predicts support/deny edge relations and samples edge reliability.

    By default, relation uncertainty is the normalized entropy of the two-class
    distribution. A differentiable Binary Concrete sample decides how much of
    the edge remains reliable before its mass is divided between the support
    and deny semantic views.

    When Dempster-Shafer mass routing is enabled, the edge head instead emits
    non-negative evidence for support/deny. The residual mass is assigned to
    the unknown set and the semantic views receive only support/deny masses.

    When Dirichlet relation routing is enabled, the edge head emits positive
    concentration parameters for a support/deny probability distribution. The
    Dirichlet mean, or an optional reparameterized Dirichlet sample, is used as
    the semantic routing probability. Its total concentration is not treated as
    edge uncertainty here; uncertainty keeps the existing entropy-based route
    ambiguity semantics.
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
        self.use_uncertainty_sampling = bool(
            getattr(args, "use_uncertainty_sampling", True)
        )
        relation_distribution = getattr(args, "edge_relation_distribution", None)
        use_dirichlet = bool(
            getattr(args, "use_dirichlet_relation_routing", False)
        )
        use_ds = bool(getattr(args, "use_ds_mass_routing", False))
        if relation_distribution is not None:
            relation_distribution = str(relation_distribution).strip().lower()
            valid_distributions = {"softmax", "ds", "dirichlet"}
            if relation_distribution not in valid_distributions:
                raise ValueError(
                    "edge_relation_distribution must be one of {}, got {}".format(
                        sorted(valid_distributions),
                        relation_distribution,
                    )
                )
            use_dirichlet = relation_distribution == "dirichlet"
            use_ds = relation_distribution == "ds"
        elif use_dirichlet and use_ds:
            raise ValueError(
                "use_dirichlet_relation_routing and use_ds_mass_routing are "
                "mutually exclusive; leave both false for the original softmax "
                "router."
            )
        self.use_dirichlet_relation_routing = use_dirichlet
        self.use_ds_mass_routing = use_ds
        self.edge_relation_distribution = (
            "dirichlet"
            if self.use_dirichlet_relation_routing
            else "ds"
            if self.use_ds_mass_routing
            else "softmax"
        )
        self.dirichlet_relation_prior = max(
            1e-6,
            float(getattr(args, "dirichlet_relation_prior", 1.0)),
        )
        self.dirichlet_relation_sample = bool(
            getattr(args, "dirichlet_relation_sample", False)
        )
        self.dirichlet_teacher_strength = max(
            1e-6,
            float(getattr(args, "dirichlet_teacher_strength", 10.0)),
        )
        self.dirichlet_teacher_smoothing = min(
            max(float(getattr(args, "dirichlet_teacher_smoothing", 0.05)), 0.0),
            0.49,
        )
        self.ds_unknown_prior = max(
            1e-6,
            float(getattr(args, "ds_unknown_prior", 2.0)),
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

    def relation_masses(self, logits):
        if logits.numel() == 0:
            empty = logits.new_zeros((0,))
            return logits.new_zeros((0, 3)), logits, empty

        evidence = F.softplus(logits / self.relation_temperature)
        total_evidence = evidence.sum(dim=-1, keepdim=True)
        denominator = total_evidence + self.ds_unknown_prior
        class_mass = evidence / denominator.clamp_min(self.eps)
        unknown_mass = (
            self.ds_unknown_prior
            / denominator.squeeze(-1).clamp_min(self.eps)
        )
        masses = torch.stack(
            (
                class_mass[:, 0],
                unknown_mass,
                class_mass[:, 1],
            ),
            dim=-1,
        )
        return masses, class_mass, unknown_mass

    def dirichlet_concentration(self, logits):
        if logits.numel() == 0:
            return logits.new_zeros(logits.size())
        evidence = F.softplus(logits / self.relation_temperature)
        return evidence + self.dirichlet_relation_prior

    def dirichlet_mean(self, concentration):
        total = concentration.sum(dim=-1, keepdim=True)
        return concentration / total.clamp_min(self.eps)

    def dirichlet_relation_probabilities(self, logits):
        concentration = self.dirichlet_concentration(logits)
        mean = self.dirichlet_mean(concentration)
        if self.training and self.dirichlet_relation_sample:
            probabilities = Dirichlet(
                concentration.clamp_min(self.eps)
            ).rsample()
            probabilities = probabilities / probabilities.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
            return concentration, probabilities
        return concentration, mean

    def dirichlet_teacher_concentration(self, labels):
        labels = labels.view(-1).long().clamp(0, 1)
        support_target = (labels == 0).to(dtype=torch.float32)
        smooth = self.dirichlet_teacher_smoothing
        support_target = support_target * (1.0 - 2.0 * smooth) + smooth
        target_prob = torch.stack(
            (support_target, 1.0 - support_target),
            dim=-1,
        )
        return (
            self.dirichlet_relation_prior
            + self.dirichlet_teacher_strength * target_prob
        )

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
        route_logits = logits
        if self.use_dirichlet_relation_routing:
            route_logits = probabilities.clamp_min(self.eps).log()
        if self.training:
            return F.gumbel_softmax(
                route_logits,
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


####这里根据边的特征构造边stance语义属于支持or反对的概率，以及不确定性，现在时ds方法建模，原始为softmax方法
    def relation_outputs(
        self,
        node_hidden,
        edge_index,
    ):
        if edge_index.numel() == 0:
            empty_logits = node_hidden.new_zeros((0, 2))
            empty_scalar = node_hidden.new_zeros((0,))
            return (
                empty_logits,
                empty_logits,
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
        if self.use_dirichlet_relation_routing:
            _, probabilities = self.dirichlet_relation_probabilities(logits)
            uncertainty = self.normalized_entropy(probabilities)
        elif self.use_ds_mass_routing:
            _, class_mass, uncertainty = self.relation_masses(logits)
            known_mass = class_mass.sum(dim=-1, keepdim=True)
            probabilities = class_mass / known_mass.clamp_min(self.eps)
        else:
            probabilities = self.relation_probabilities(logits)
            uncertainty = self.normalized_entropy(probabilities)
        return logits, probabilities, uncertainty

    def route_edges(
        self,
        logits,
        probabilities,
        uncertainty,
        child_degree_importance=None,
    ):
        if logits.numel() == 0:
            empty_scalar = uncertainty.new_zeros((0,))
            return empty_scalar, empty_scalar, empty_scalar

        if self.use_ds_mass_routing:
            known_mass = (1.0 - uncertainty).clamp(0.0, 1.0)
            support_weight = known_mass * probabilities[:, 0]
            deny_weight = known_mass * probabilities[:, 1]
            return known_mass, support_weight, deny_weight
        #这里目前设置了use_uncertainty_sampling为false，所以不进行伯努利采样
        if (
            not self.use_uncertainty_sampling
            or self.current_epoch < self.warmup_epochs
        ):
            keep_probability = uncertainty.new_ones(uncertainty.size())
            keep_sample = torch.ones_like(keep_probability)
        else:
            keep_probability = self.reliability_probability(
                uncertainty,
                child_degree_importance,
            )
            keep_sample = self.soft_bernoulli_sample(keep_probability)


        # Two-stage routing:
        #   1. keep_sample decides whether the uncertain edge is retained;
        #   2. stance_route decides which semantic view receives it.
        route = self.stance_route(logits, probabilities)
        support_weight = keep_sample * route[:, 0]
        deny_weight = keep_sample * route[:, 1]
        return keep_sample, support_weight, deny_weight

    def forward(
        self,
        node_hidden,
        edge_index,
        child_degree_importance=None,
    ):
        logits, probabilities, uncertainty = self.relation_outputs(
            node_hidden,
            edge_index,
        )
        keep_sample, support_weight, deny_weight = self.route_edges(
            logits,
            probabilities,
            uncertainty,
            child_degree_importance,
        )
        return (
            logits,
            probabilities,
            uncertainty,
            keep_sample,
            support_weight,
            deny_weight,
        )


class SemanticParityDirectionEncoder(nn.Module):
    """
    Propagates support/deny semantics as composable path parity.

    Support edges preserve the current semantic channel, while deny edges swap
    the support and deny channels. Stacking this layer therefore keeps the
    usual conflict algebra valid for arbitrary hop counts:

        support + deny = deny
        deny + deny = support
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
        residual=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.dropout = float(dropout)
        self.residual = bool(residual)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )

    def forward(self, node_features, edge_index, support_weight, deny_weight):
        same = self.input_projection(node_features.float())
        diff = same.new_zeros(same.size())

        for layer, norm in zip(self.layers, self.norms):
            same_aggr, diff_aggr = self._aggregate_parity(
                same,
                diff,
                edge_index,
                support_weight,
                deny_weight,
            )
            same_update = F.relu(norm(layer(same_aggr)))
            diff_update = F.relu(norm(layer(diff_aggr)))
            same_update = F.dropout(
                same_update,
                p=self.dropout,
                training=self.training,
            )
            diff_update = F.dropout(
                diff_update,
                p=self.dropout,
                training=self.training,
            )
            if self.residual:
                same = same + same_update
                diff = diff + diff_update
            else:
                same = same_update
                diff = diff_update
        return same, diff

    def _aggregate_parity(
        self,
        same,
        diff,
        edge_index,
        support_weight,
        deny_weight,
    ):
        same_out = same.clone()
        diff_out = diff.clone()
        denom = same.new_ones(same.size(0), 1)
        if edge_index.numel() == 0:
            return same_out, diff_out

        src, dst = edge_index
        support = support_weight.to(dtype=same.dtype).view(-1, 1)
        deny = deny_weight.to(dtype=same.dtype).view(-1, 1)

        same_msg = support * same[src] + deny * diff[src]
        diff_msg = support * diff[src] + deny * same[src]
        edge_mass = (support + deny).clamp_min(0.0)

        same_out.index_add_(0, dst, same_msg)
        diff_out.index_add_(0, dst, diff_msg)
        denom.index_add_(0, dst, edge_mass)
        return same_out / denom.clamp_min(1e-6), diff_out / denom.clamp_min(
            1e-6
        )


class SemanticParityEncoder(nn.Module):
    """
    Support/deny view encoder with optional bidirectional tree propagation.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
        bidirectional=False,
        residual=True,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.top_down = SemanticParityDirectionEncoder(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            residual=residual,
        )
        if self.bidirectional:
            self.bottom_up = SemanticParityDirectionEncoder(
                input_dim,
                hidden_dim,
                num_layers,
                dropout=dropout,
                residual=residual,
            )
            self.direction_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
                nn.ReLU(),
            )
        else:
            self.bottom_up = None
            self.direction_fusion = None

    def forward(self, node_features, edge_index, support_weight, deny_weight):
        same_td, diff_td = self.top_down(
            node_features,
            edge_index,
            support_weight,
            deny_weight,
        )
        if not self.bidirectional:
            return same_td, diff_td

        reverse_edge_index = torch.stack(
            (edge_index[1], edge_index[0]),
            dim=0,
        )
        same_bu, diff_bu = self.bottom_up(
            node_features,
            reverse_edge_index,
            support_weight,
            deny_weight,
        )
        same = self.direction_fusion(torch.cat((same_td, same_bu), dim=-1))
        diff = self.direction_fusion(torch.cat((diff_td, diff_bu), dim=-1))
        return same, diff


class GlobalDSFusionHead(nn.Module):
    """
    Graph-level Dempster-Shafer fusion over classification branches.

    Each branch emits singleton class masses plus one full-frame unknown mass.
    The branch masses are then combined by Dempster's rule, or by Yager's rule
    when high conflict should remain unknown instead of being normalized away.
    """

    def __init__(self, hidden_dim, num_classes, branch_names, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("GlobalDSFusionHead requires at least 2 classes")
        self.branch_names = tuple(branch_names)
        self.temperature = max(
            1e-6,
            float(getattr(args, "global_ds_temperature", 1.0)),
        )
        self.unknown_prior = max(
            1e-6,
            float(getattr(args, "global_ds_unknown_prior", 1.0)),
        )
        self.eps = max(
            1e-12,
            float(getattr(args, "global_ds_eps", 1e-6)),
        )
        self.fusion_rule = str(
            getattr(args, "global_ds_fusion_rule", "dempster")
        ).strip().lower()
        if self.fusion_rule not in {"dempster", "yager"}:
            raise ValueError(
                "global_ds_fusion_rule must be 'dempster' or 'yager', "
                "got {}".format(self.fusion_rule)
            )
        head_hidden = int(
            getattr(args, "global_ds_hidden_dim", hidden_dim)
        )
        self.mass_heads = nn.ModuleDict()
        for branch_name in self.branch_names:
            self.mass_heads[branch_name] = nn.Sequential(
                nn.Linear(self.hidden_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(float(getattr(args, "dropout", 0.0))),
                nn.Linear(head_hidden, self.num_classes),
            )

    def branch_mass(self, branch_name, graph_repr):
        logits = self.mass_heads[branch_name](graph_repr)
        evidence = F.softplus(logits / self.temperature)
        total_evidence = evidence.sum(dim=-1, keepdim=True)
        denominator = total_evidence + self.unknown_prior
        class_mass = evidence / denominator.clamp_min(self.eps)
        unknown_mass = (
            self.unknown_prior / denominator.clamp_min(self.eps)
        )
        masses = torch.cat((class_mass, unknown_mass), dim=-1)
        return masses

    def combine_pair(self, first, second):
        first_class = first[:, : self.num_classes]
        second_class = second[:, : self.num_classes]
        first_unknown = first[:, self.num_classes :]
        second_unknown = second[:, self.num_classes :]

        agreement = first_class * second_class
        class_numerator = (
            agreement
            + first_class * second_unknown
            + first_unknown * second_class
        )
        unknown_numerator = first_unknown * second_unknown
        conflict = (
            first_class.sum(dim=-1, keepdim=True)
            * second_class.sum(dim=-1, keepdim=True)
            - agreement.sum(dim=-1, keepdim=True)
        ).clamp_min(0.0)

        if self.fusion_rule == "yager":
            return torch.cat(
                (
                    class_numerator,
                    unknown_numerator + conflict,
                ),
                dim=-1,
            ), conflict

        normalizer = (1.0 - conflict).clamp_min(self.eps)
        return torch.cat(
            (
                class_numerator / normalizer,
                unknown_numerator / normalizer,
            ),
            dim=-1,
        ), conflict

    def pignistic_probabilities(self, masses):
        class_mass = masses[:, : self.num_classes]
        unknown_mass = masses[:, self.num_classes :]
        probabilities = class_mass + unknown_mass / float(self.num_classes)
        probabilities = probabilities.clamp_min(self.eps)
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    def forward(self, graph_branches):
        branch_masses = []
        for branch_name in self.branch_names:
            branch_masses.append(
                self.branch_mass(branch_name, graph_branches[branch_name])
            )
        stacked_branch_masses = torch.stack(branch_masses, dim=1)
        combined = branch_masses[0]
        conflicts = []
        for branch_mass in branch_masses[1:]:
            combined, conflict = self.combine_pair(combined, branch_mass)
            conflicts.append(conflict.squeeze(-1))
        probabilities = self.pignistic_probabilities(combined)
        if conflicts:
            conflict_trace = torch.stack(conflicts, dim=1)
        else:
            conflict_trace = combined.new_zeros((combined.size(0), 0))
        return (
            probabilities.log(),
            combined,
            stacked_branch_masses,
            conflict_trace,
        )


class RootPathUncertaintyAttention(nn.Module):
    """
    Updates each non-root node from the nodes on its root path.

    For query node u and path node v, the attention bias is controlled by
    distance(u, v), the learned uncertainty of v, and their interaction. The
    uncertainty is local node uncertainty: root is 0, and every other node uses
    the entropy of its incoming parent edge.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = max(
            1,
            int(getattr(args, "vertical_path_attention_heads", 4)),
        )
        if self.hidden_dim % self.heads != 0:
            raise ValueError(
                "hidden_dim {} must be divisible by "
                "vertical_path_attention_heads {}".format(
                    self.hidden_dim,
                    self.heads,
                )
            )
        self.head_dim = self.hidden_dim // self.heads
        self.score_scale = self.head_dim ** -0.5
        self.max_distance = max(
            0,
            int(
                getattr(
                    args,
                    "vertical_path_attention_max_distance",
                    getattr(args, "max_hop", 32),
                )
            ),
        )
        dropout = float(
            getattr(
                args,
                "vertical_path_attention_dropout",
                getattr(args, "dropout", 0.0),
            )
        )

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.distance_bias = nn.Embedding(
            self.max_distance + 1,
            self.heads,
        )
        self.distance_uncertainty_bias = nn.Embedding(
            self.max_distance + 1,
            self.heads,
        )
        nn.init.zeros_(self.distance_bias.weight)
        nn.init.zeros_(self.distance_uncertainty_bias.weight)

        uncertainty_scale = max(
            1e-6,
            float(
                getattr(
                    args,
                    "vertical_path_attention_uncertainty_scale",
                    1.0,
                )
            ),
        )
        if uncertainty_scale > 20.0:
            uncertainty_raw = uncertainty_scale
        else:
            uncertainty_raw = math.log(math.expm1(uncertainty_scale))
        self.uncertainty_scale_raw = nn.Parameter(
            torch.full((self.heads,), uncertainty_raw)
        )

        gate = float(
            getattr(args, "vertical_path_attention_residual_gate", 1.0)
        )
        gate = min(max(gate, 1e-4), 1.0 - 1e-4)
        self.residual_gate_raw = nn.Parameter(
            torch.tensor(math.log(gate / (1.0 - gate)))
        )

    def forward(
        self,
        node_hidden,
        parent,
        depth,
        node_uncertainty,
    ):
        query_index, key_index, distance = self._build_path_pairs(
            parent,
            depth,
        )
        if query_index.numel() == 0:
            return node_hidden

        num_nodes = node_hidden.size(0)
        query = self.q_proj(node_hidden).view(
            num_nodes,
            self.heads,
            self.head_dim,
        )
        key = self.k_proj(node_hidden).view(
            num_nodes,
            self.heads,
            self.head_dim,
        )
        value = self.v_proj(node_hidden).view(
            num_nodes,
            self.heads,
            self.head_dim,
        )

        attention_score = (
            query[query_index] * key[key_index]
        ).sum(dim=-1) * self.score_scale
        distance = distance.clamp(0, self.max_distance)
        key_uncertainty = node_uncertainty[key_index].clamp(
            0.0,
            1.0,
        )
        distance_bias = self.distance_bias(distance)
        distance_uncertainty_bias = (
            self.distance_uncertainty_bias(distance)
            * key_uncertainty.unsqueeze(-1)
        )
        uncertainty_penalty = (
            F.softplus(self.uncertainty_scale_raw).view(1, -1)
            * key_uncertainty.unsqueeze(-1)
        )
        attention_score = (
            attention_score
            + distance_bias
            + distance_uncertainty_bias
            - uncertainty_penalty
        )

        attention = softmax(
            attention_score,
            query_index,
            num_nodes=num_nodes,
        )
        attention = self.attention_dropout(attention)
        message = attention.unsqueeze(-1) * value[key_index]

        path_hidden = node_hidden.new_zeros(
            num_nodes,
            self.heads,
            self.head_dim,
        )
        path_hidden.index_add_(0, query_index, message)
        path_hidden = path_hidden.reshape(num_nodes, self.hidden_dim)
        path_hidden = self.out_proj(path_hidden)
        path_hidden = self.output_dropout(path_hidden)

        updated = node_hidden.clone()
        query_nodes = query_index.unique()
        gate = torch.sigmoid(self.residual_gate_raw)
        updated[query_nodes] = self.norm(
            node_hidden[query_nodes] + gate * path_hidden[query_nodes]
        )
        return updated

    def _build_path_pairs(self, parent, depth):
        query_nodes = (depth > 0).nonzero(as_tuple=False).view(-1)
        if query_nodes.numel() == 0:
            empty = parent.new_zeros((0,))
            return empty, empty, empty

        max_steps = int(depth[query_nodes].max().item()) + 1
        current = query_nodes.clone()
        query_parts = []
        key_parts = []
        distance_parts = []
        for _ in range(max_steps):
            active = current >= 0
            if not active.any():
                break
            active_query = query_nodes[active]
            active_key = current[active]
            valid_depth = depth[active_key] >= 0
            if valid_depth.any():
                active_query = active_query[valid_depth]
                active_key = active_key[valid_depth]
                query_parts.append(active_query)
                key_parts.append(active_key)
                distance_parts.append(
                    (depth[active_query] - depth[active_key])
                    .clamp_min(0)
                    .long()
                )

            next_current = current.new_full(current.size(), -1)
            next_current[active] = parent[current[active]]
            current = next_current

        if not query_parts:
            empty = parent.new_zeros((0,))
            return empty, empty, empty
        return (
            torch.cat(query_parts, dim=0),
            torch.cat(key_parts, dim=0),
            torch.cat(distance_parts, dim=0),
        )


class _TopicGuidedValuePropagationLayer(nn.Module):
    """Propagates one value view over a fixed topic-attention graph."""

    def __init__(self, hidden_dim, feedforward_dim, dropout):
        super().__init__()
        self.value_norm = nn.LayerNorm(hidden_dim)
        self.value_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, value, attention, valid_mask):
        context = torch.matmul(attention, self.value_norm(value))
        value = value + self.context_dropout(self.value_output(context))
        value = value + self.feedforward(self.ffn_norm(value))
        return value.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class SemanticTreeTransformerBranch(nn.Module):
    """Topic-guided dual-view propagation over each propagation tree.

    The original propagation-graph node representations are used only to
    infer a soft topic distribution and a symmetric pairwise topic affinity.
    Row-wise normalization turns that shared pair score into the aggregation
    attention.  The values come from the support/deny graph views together
    with absolute depth, and both views reuse exactly the same attention.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_depth = max(
            0,
            int(
                getattr(
                    args,
                    "semantic_tree_transformer_max_depth",
                    getattr(args, "max_hop", 32),
                )
            ),
        )
        depth_dim = max(
            1,
            int(
                getattr(
                    args,
                    "semantic_tree_depth_dim",
                    self.hidden_dim,
                )
            ),
        )
        legacy_heads = max(
            1,
            int(getattr(args, "semantic_tree_transformer_heads", 4)),
        )
        self.num_topics = max(
            1,
            int(getattr(args, "semantic_tree_num_topics", legacy_heads)),
        )
        layers = max(
            1,
            int(getattr(args, "semantic_tree_transformer_layers", 1)),
        )
        feedforward_dim = max(
            self.hidden_dim,
            int(
                getattr(
                    args,
                    "semantic_tree_transformer_ffn_dim",
                    self.hidden_dim * 2,
                )
            ),
        )
        dropout = float(
            getattr(
                args,
                "semantic_tree_transformer_dropout",
                getattr(args, "dropout", 0.0),
            )
        )
        self.topic_temperature = max(
            1e-6,
            float(getattr(args, "semantic_tree_topic_temperature", 1.0)),
        )
        self.attention_temperature = max(
            1e-6,
            float(
                getattr(
                    args,
                    "semantic_tree_attention_temperature",
                    1.0,
                )
            ),
        )
        self.topic_eps = max(
            1e-12,
            float(getattr(args, "semantic_tree_topic_eps", 1e-6)),
        )
        self.pool = str(
            getattr(args, "semantic_tree_transformer_pool", "mean")
        ).strip().lower()
        if self.pool not in {"mean", "sum", "root"}:
            raise ValueError(
                "semantic_tree_transformer_pool must be one of "
                "['mean', 'root', 'sum'], got {}".format(self.pool)
            )

        self.depth_embedding = nn.Embedding(self.max_depth + 2, depth_dim)
        self.support_missing = nn.Parameter(torch.zeros(self.hidden_dim))
        self.deny_missing = nn.Parameter(torch.zeros(self.hidden_dim))
        self.topic_encoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_topics),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(self.hidden_dim + depth_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.propagation_layers = nn.ModuleList(
            [
                _TopicGuidedValuePropagationLayer(
                    self.hidden_dim,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.view_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )

        self.last_topic_distribution = None
        self.last_topic_similarity = None
        self.last_attention = None

    def _inject_missing_view(self, view_nodes, node_weight, missing):
        if node_weight is None:
            return view_nodes
        weight = node_weight.to(dtype=view_nodes.dtype).clamp(0.0, 1.0)
        weight = weight.unsqueeze(-1)
        return view_nodes * weight + missing.view(1, -1) * (1.0 - weight)

    def _depth_indices(self, depth):
        # Unknown/unreachable depth -1 maps to 0. Root depth 0 maps to 1.
        return depth.long().clamp(-1, self.max_depth) + 1

    def _topic_attention(self, original_dense, valid_mask):
        topic_logits = self.topic_encoder(original_dense)
        topic_distribution = F.softmax(
            topic_logits / self.topic_temperature,
            dim=-1,
        )
        topic_similarity = torch.matmul(
            topic_distribution,
            topic_distribution.transpose(1, 2),
        )
        # log shared-topic mass has a useful range: identical sharp topics are
        # near zero, while disjoint topics receive a large negative logit.
        score = torch.log(topic_similarity.clamp_min(self.topic_eps))
        score = score / self.attention_temperature
        score = score.masked_fill(~valid_mask.unsqueeze(1), -1e9)
        attention = F.softmax(score, dim=-1)
        attention = self.attention_dropout(attention)
        attention = attention.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )
        return topic_distribution, topic_similarity, attention

    def forward(
        self,
        original_nodes,
        support_nodes,
        deny_nodes,
        depth,
        batch,
        support_node_weight=None,
        deny_node_weight=None,
    ):
        support_nodes = self._inject_missing_view(
            support_nodes,
            support_node_weight,
            self.support_missing,
        )
        deny_nodes = self._inject_missing_view(
            deny_nodes,
            deny_node_weight,
            self.deny_missing,
        )
        depth_nodes = self.depth_embedding(self._depth_indices(depth))

        original_dense, valid_mask = to_dense_batch(original_nodes, batch)
        support_dense, support_mask = to_dense_batch(
            self.value_projection(
                torch.cat((support_nodes, depth_nodes), dim=-1)
            ),
            batch,
        )
        deny_dense, deny_mask = to_dense_batch(
            self.value_projection(
                torch.cat((deny_nodes, depth_nodes), dim=-1)
            ),
            batch,
        )
        valid_mask = valid_mask & support_mask & deny_mask
        (
            topic_distribution,
            topic_similarity,
            attention,
        ) = self._topic_attention(original_dense, valid_mask)

        for layer in self.propagation_layers:
            support_dense = layer(support_dense, attention, valid_mask)
            deny_dense = layer(deny_dense, attention, valid_mask)

        delta = deny_dense - support_dense
        encoded_dense = self.view_fusion(
            torch.cat(
                (support_dense, deny_dense, delta, delta.abs()),
                dim=-1,
            )
        )
        encoded_dense = encoded_dense.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )

        if self.pool == "root":
            graph_hidden = encoded_dense[:, 0]
        else:
            mask = valid_mask.unsqueeze(-1).to(dtype=encoded_dense.dtype)
            graph_hidden = (encoded_dense * mask).sum(dim=1)
            if self.pool == "mean":
                graph_hidden = graph_hidden / mask.sum(dim=1).clamp_min(1.0)

        self.last_topic_distribution = topic_distribution
        self.last_topic_similarity = topic_similarity
        self.last_attention = attention
        encoded_nodes = encoded_dense[valid_mask]
        return graph_hidden, encoded_nodes


class ConflictConditionedTransformerBlock(nn.Module):
    """Pre-norm Transformer block with an additive conflict attention bias."""

    def __init__(self, hidden_dim, heads, feedforward_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden, valid_mask, attention_bias):
        normalized = self.norm1(hidden)
        padding_mask = torch.zeros(
            valid_mask.size(),
            dtype=normalized.dtype,
            device=normalized.device,
        )
        padding_mask = padding_mask.masked_fill(
            ~valid_mask,
            float("-inf"),
        )
        attended, attention = self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_bias,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.feedforward(self.norm2(hidden))
        hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return hidden, attention


class ConflictConditionedTransformer(nn.Module):
    """
    Uses a dual-view conflict field as an explicit Transformer prior.

    Attention heads cycle through four interpretable roles:
      * key: attend to nodes with high conflict intensity;
      * boundary: attend across sharp conflict-intensity transitions;
      * region: model interactions inside high-conflict regions;
      * free: retain unconstrained content-based attention.

    Unlike the legacy conflict pooling branch, this encoder is deterministic
    and lets conflict scores affect node-to-node information exchange before
    graph readout.
    """

    HEAD_ROLES = ("key", "boundary", "region", "free")

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = max(
            1,
            int(getattr(args, "conflict_attention_heads", 4)),
        )
        if self.hidden_dim % self.heads != 0:
            raise ValueError(
                "hidden_dim {} must be divisible by "
                "conflict_attention_heads {}".format(
                    self.hidden_dim,
                    self.heads,
                )
            )
        layers = max(
            1,
            int(getattr(args, "conflict_attention_layers", 1)),
        )
        feedforward_dim = max(
            self.hidden_dim,
            int(
                getattr(
                    args,
                    "conflict_attention_ffn_dim",
                    self.hidden_dim * 2,
                )
            ),
        )
        dropout = float(
            getattr(
                args,
                "conflict_attention_dropout",
                getattr(args, "dropout", 0.0),
            )
        )
        self.max_depth = max(
            1,
            int(
                getattr(
                    args,
                    "conflict_attention_max_depth",
                    getattr(args, "max_hop", 32),
                )
            ),
        )
        depth_dim = max(
            1,
            int(
                getattr(
                    args,
                    "conflict_attention_depth_dim",
                    self.hidden_dim,
                )
            ),
        )
        self.pool = str(
            getattr(args, "conflict_attention_pool", "mean")
        ).strip().lower()
        if self.pool not in {"mean", "root", "sum"}:
            raise ValueError(
                "conflict_attention_pool must be one of "
                "['mean', 'root', 'sum'], got {}".format(self.pool)
            )

        self.head_roles = tuple(
            self.HEAD_ROLES[index % len(self.HEAD_ROLES)]
            for index in range(self.heads)
        )
        role_scales = {
            "key": float(
                getattr(args, "conflict_attention_key_scale", 1.0)
            ),
            "boundary": float(
                getattr(args, "conflict_attention_boundary_scale", 1.0)
            ),
            "region": float(
                getattr(args, "conflict_attention_region_scale", 1.0)
            ),
        }
        for role, value in role_scales.items():
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(
                    "conflict attention {} scale must be finite and "
                    "non-negative, got {}".format(role, value)
                )
        self.key_scale_raw = nn.Parameter(
            self._inverse_softplus(role_scales["key"])
        )
        self.boundary_scale_raw = nn.Parameter(
            self._inverse_softplus(role_scales["boundary"])
        )
        self.region_scale_raw = nn.Parameter(
            self._inverse_softplus(role_scales["region"])
        )

        self.depth_embedding = nn.Embedding(self.max_depth + 2, depth_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 3 + depth_dim + 4,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ConflictConditionedTransformerBlock(
                    self.hidden_dim,
                    self.heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def _inverse_softplus(value):
        value = max(float(value), 1e-6)
        return torch.tensor(math.log(math.expm1(value)))

    def _depth_indices(self, depth):
        return depth.long().clamp(-1, self.max_depth) + 1

    def attention_scales(self):
        return {
            "key": F.softplus(self.key_scale_raw),
            "boundary": F.softplus(self.boundary_scale_raw),
            "region": F.softplus(self.region_scale_raw),
        }

    def build_attention_bias(self, dense_score, valid_mask):
        score = dense_score.clamp(0.0, 1.0)
        query_score = score.unsqueeze(-1)
        key_score = score.unsqueeze(-2)
        role_bias = {
            "key": key_score.expand(-1, score.size(1), -1),
            "boundary": (query_score - key_score).abs(),
            "region": query_score * key_score,
            "free": torch.zeros(
                score.size(0),
                score.size(1),
                score.size(1),
                dtype=score.dtype,
                device=score.device,
            ),
        }
        scales = self.attention_scales()
        head_biases = []
        for role in self.head_roles:
            bias = role_bias[role]
            if role in scales:
                bias = bias * scales[role].to(dtype=bias.dtype)
            head_biases.append(bias)
        attention_bias = torch.stack(head_biases, dim=1)
        valid_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        attention_bias = attention_bias * valid_pair.unsqueeze(1).to(
            dtype=attention_bias.dtype
        )
        return attention_bias.reshape(
            score.size(0) * self.heads,
            score.size(1),
            score.size(1),
        )

    def _attention_received(self, attention, valid_mask):
        query_mask = valid_mask[:, None, :, None].to(
            dtype=attention.dtype
        )
        query_count = query_mask.sum(dim=2).clamp_min(1.0)
        received = (attention * query_mask).sum(dim=2) / query_count
        received = received * valid_mask[:, None, :].to(
            dtype=received.dtype
        )
        by_head = received.permute(0, 2, 1)[valid_mask]
        return by_head.mean(dim=-1), by_head

    def forward(
        self,
        original_nodes,
        support_nodes,
        deny_nodes,
        field,
        conflict_intensity,
        high_frequency,
        conflict_score,
        depth,
        batch,
    ):
        scalar_features = torch.stack(
            (
                field,
                conflict_intensity,
                high_frequency,
                conflict_score,
            ),
            dim=-1,
        )
        depth_features = self.depth_embedding(self._depth_indices(depth))
        node_input = torch.cat(
            (
                original_nodes,
                support_nodes,
                deny_nodes,
                depth_features,
                scalar_features,
            ),
            dim=-1,
        )
        node_hidden = self.input_projection(node_input)
        dense_hidden, valid_mask = to_dense_batch(node_hidden, batch)
        dense_score, score_mask = to_dense_batch(conflict_score, batch)
        if not torch.equal(valid_mask, score_mask):
            raise ValueError(
                "conflict score batching is not aligned with node batching"
            )
        attention_bias = self.build_attention_bias(
            dense_score,
            valid_mask,
        )
        attention = None
        for block in self.blocks:
            dense_hidden, attention = block(
                dense_hidden,
                valid_mask,
                attention_bias,
            )
        dense_hidden = self.output_norm(dense_hidden)
        dense_hidden = dense_hidden.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )

        if self.pool == "root":
            graph_hidden = dense_hidden[:, 0]
        else:
            mask = valid_mask.unsqueeze(-1).to(dtype=dense_hidden.dtype)
            graph_hidden = (dense_hidden * mask).sum(dim=1)
            if self.pool == "mean":
                graph_hidden = graph_hidden / mask.sum(dim=1).clamp_min(1.0)

        encoded_nodes = dense_hidden[valid_mask]
        attention_received, attention_by_head = self._attention_received(
            attention,
            valid_mask,
        )
        node_importance = attention_received * conflict_score
        return (
            graph_hidden,
            encoded_nodes,
            attention_received,
            attention_by_head,
            node_importance,
        )


class ConflictFieldBottleneck(nn.Module):
    """
    Builds a signed support-deny semantic field and encodes high-conflict
    propagation evidence.

    The scalar field is close to 0 when support and deny semantics are
    balanced, and close to +/-1 when one side dominates. ``pool`` mode keeps
    the legacy Binary-Concrete weighted readout. ``transformer`` mode uses the
    conflict score as a deterministic multi-head attention prior, so key
    nodes, conflict boundaries, and high-conflict regions affect information
    exchange before graph readout.
    """

    def __init__(self, hidden_dim, num_classes, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder_mode = str(
            getattr(args, "conflict_encoder_mode", "pool")
        ).strip().lower()
        if self.encoder_mode not in {"pool", "residual", "transformer"}:
            raise ValueError(
                "conflict_encoder_mode must be one of "
                "['pool', 'residual', 'transformer'], got {}".format(
                    self.encoder_mode
                )
            )
        field_hidden = int(
            getattr(args, "conflict_field_hidden_dim", hidden_dim)
        )
        self.field_temperature = max(
            1e-6,
            float(getattr(args, "conflict_field_temperature", 1.0)),
        )
        self.score_temperature = max(
            1e-6,
            float(getattr(args, "conflict_score_temperature", 0.2)),
        )
        self.sample_temperature = max(
            1e-6,
            float(getattr(args, "conflict_sample_temperature", 0.5)),
        )
        self.keep_threshold = float(
            getattr(args, "conflict_keep_threshold", 0.5)
        )
        self.keep_floor = min(
            max(float(getattr(args, "conflict_keep_floor", 0.05)), 0.0),
            1.0,
        )
        self.conflict_weight = max(
            0.0,
            float(getattr(args, "conflict_balance_weight", 1.0)),
        )
        self.high_frequency_weight = max(
            0.0,
            float(getattr(args, "conflict_high_frequency_weight", 0.5)),
        )
        self.use_local_high_frequency = bool(
            getattr(args, "use_conflict_local_high_frequency", True)
        )
        self.use_energy_gate = bool(
            getattr(args, "use_conflict_energy_gate", True)
        )
        self.force_root_keep = bool(
            getattr(args, "conflict_force_root_keep", True)
        )
        self.hard_sample = bool(
            getattr(args, "conflict_hard_sample", False)
        )
        self.eval_hard = bool(
            getattr(args, "conflict_eval_hard", False)
        )
        self.warmup_epochs = max(
            0,
            int(getattr(args, "conflict_sampling_warmup_epochs", 0)),
        )
        self.lambda_label = max(
            0.0,
            float(getattr(args, "lambda_conflict_label_aux", 0.0)),
        )
        self.lambda_size = max(
            0.0,
            float(getattr(args, "lambda_conflict_size_aux", 0.0)),
        )
        self.lambda_redundancy = max(
            0.0,
            float(getattr(args, "lambda_conflict_redundancy_aux", 0.0)),
        )
        self.residual_scale = max(
            0.0,
            float(getattr(args, "conflict_residual_scale", 0.5)),
        )
        self.residual_gate_floor = min(
            max(
                float(getattr(args, "conflict_residual_gate_floor", 0.25)),
                0.0,
            ),
            1.0,
        )
        self.readout_strength = min(
            max(
                float(getattr(args, "conflict_readout_strength", 0.5)),
                0.0,
            ),
            1.0,
        )
        dropout = float(getattr(args, "dropout", 0.0))
        self.eps = 1e-6
        self.register_buffer(
            "_current_epoch",
            torch.zeros((), dtype=torch.long),
        )

        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, field_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(field_hidden, 1),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, field_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(field_hidden, 1),
        )
        self.node_encoder = (
            nn.Sequential(
                nn.Linear(hidden_dim * 3 + 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            if self.encoder_mode == "pool"
            else None
        )
        self.residual_encoder = (
            nn.Sequential(
                nn.Linear(hidden_dim * 3 + 3, field_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(field_hidden, hidden_dim),
            )
            if self.encoder_mode == "residual"
            else None
        )
        self.residual_gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 4 + 3, field_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(field_hidden, 1),
            )
            if self.encoder_mode == "residual"
            else None
        )
        self.residual_norm = (
            nn.LayerNorm(hidden_dim)
            if self.encoder_mode == "residual"
            else None
        )
        self.attention_encoder = (
            ConflictConditionedTransformer(hidden_dim, args=args)
            if self.encoder_mode == "transformer"
            else None
        )
        self.label_head = nn.Linear(hidden_dim, num_classes)

    def set_epoch(self, epoch):
        self._current_epoch.fill_(max(0, int(epoch)))

    @property
    def current_epoch(self):
        return int(self._current_epoch.item())

    def soft_bernoulli_sample(self, keep_probability):
        keep_probability = keep_probability.clamp(
            self.eps,
            1.0 - self.eps,
        )
        if self.current_epoch < self.warmup_epochs:
            return torch.ones_like(keep_probability)
        if not self.training:
            if self.eval_hard:
                return (keep_probability >= self.keep_threshold).to(
                    dtype=keep_probability.dtype
                )
            return keep_probability

        uniform = torch.rand_like(keep_probability).clamp(
            self.eps,
            1.0 - self.eps,
        )
        logistic_noise = uniform.log() - torch.log1p(-uniform)
        keep_logit = torch.logit(keep_probability, eps=self.eps)
        sample = torch.sigmoid(
            (keep_logit + logistic_noise) / self.sample_temperature
        )
        if self.hard_sample:
            hard = (sample >= 0.5).to(dtype=sample.dtype)
            sample = hard.detach() - sample.detach() + sample
        return sample

    def _local_high_frequency(self, field, edge_index):
        if edge_index.numel() == 0:
            return torch.zeros_like(field)
        src, dst = edge_index
        edge_variation = (field[src] - field[dst]).abs() * 0.5
        variation_sum = torch.zeros_like(field)
        variation_count = torch.zeros_like(field)
        variation_sum.index_add_(0, src, edge_variation)
        variation_sum.index_add_(0, dst, edge_variation)
        ones = torch.ones_like(edge_variation)
        variation_count.index_add_(0, src, ones)
        variation_count.index_add_(0, dst, ones)
        return variation_sum / variation_count.clamp_min(1.0)

    def _pool_nodes(self, node_hidden, node_weight, batch, pool_is_sum):
        weight = node_weight.unsqueeze(-1).to(dtype=node_hidden.dtype)
        weighted_sum = global_add_pool(node_hidden * weight, batch)
        if pool_is_sum:
            return weighted_sum
        weight_sum = global_add_pool(weight, batch)
        return weighted_sum / weight_sum.clamp_min(1e-6)

    def _regularizer_mask(self, num_nodes, roots, device, dtype):
        mask = torch.ones(num_nodes, device=device, dtype=dtype)
        if roots is not None and roots.numel() > 0:
            mask[roots] = 0.0
        return mask

    def _masked_mean(self, value, mask):
        if value.numel() == 0:
            return value.new_zeros(())
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def forward(
        self,
        support_nodes,
        deny_nodes,
        change_nodes,
        edge_index,
        batch,
        roots,
        pool_is_sum=False,
        original_graph=None,
        target=None,
        base_node_keep=None,
        original_nodes=None,
        depth=None,
    ):
        delta = support_nodes - deny_nodes
        abs_delta = delta.abs()
        support_strength = self.strength_head(support_nodes).squeeze(-1)
        deny_strength = self.strength_head(deny_nodes).squeeze(-1)
        field = torch.tanh(
            (support_strength - deny_strength) / self.field_temperature
        )

        balance_conflict = (1.0 - field.abs()).clamp(0.0, 1.0)
        energy_input = torch.cat(
            (support_nodes, deny_nodes, delta, abs_delta),
            dim=-1,
        )
        if self.use_energy_gate:
            evidence_energy = torch.sigmoid(
                self.energy_head(energy_input).squeeze(-1)
            )
        else:
            evidence_energy = torch.ones_like(balance_conflict)
        conflict_intensity = balance_conflict * evidence_energy

        if self.use_local_high_frequency:
            high_frequency = self._local_high_frequency(field, edge_index)
        else:
            high_frequency = torch.zeros_like(conflict_intensity)

        score_denominator = (
            self.conflict_weight
            + (
                self.high_frequency_weight
                if self.use_local_high_frequency
                else 0.0
            )
        )
        if score_denominator <= 0.0:
            score_denominator = 1.0
        conflict_score = (
            self.conflict_weight * conflict_intensity
            + self.high_frequency_weight * high_frequency
        ) / score_denominator
        keep_probability = torch.sigmoid(
            (conflict_score - self.keep_threshold)
            / self.score_temperature
        )
        keep_probability = (
            self.keep_floor
            + (1.0 - self.keep_floor) * keep_probability
        )
        if base_node_keep is not None:
            keep_probability = keep_probability * base_node_keep.clamp(
                0.0,
                1.0,
            )
        if self.force_root_keep and roots is not None and roots.numel() > 0:
            keep_probability = keep_probability.clone()
            keep_probability[roots] = 1.0

        attention_received = None
        attention_by_head = None
        node_importance = None
        node_gate = None
        if self.encoder_mode == "transformer":
            # The Transformer consumes the pure conflict score before DS/path
            # reliability attenuation. This preserves the interpretation that
            # a higher semantic conflict score always supplies a stronger
            # conflict-attention prior.
            keep_sample = keep_probability
            if original_nodes is None:
                raise ValueError(
                    "original_nodes is required when "
                    "conflict_encoder_mode='transformer'"
                )
            if depth is None:
                raise ValueError(
                    "depth is required when "
                    "conflict_encoder_mode='transformer'"
                )
            (
                graph_hidden,
                conflict_nodes,
                attention_received,
                attention_by_head,
                node_importance,
            ) = self.attention_encoder(
                original_nodes,
                support_nodes,
                deny_nodes,
                field,
                conflict_intensity,
                high_frequency,
                conflict_score,
                depth,
                batch,
            )
        elif self.encoder_mode == "residual":
            if change_nodes is None:
                raise ValueError(
                    "change_nodes is required when "
                    "conflict_encoder_mode='residual'"
                )
            scalar_features = torch.stack(
                (field, conflict_intensity, high_frequency),
                dim=-1,
            )
            residual_input = torch.cat(
                (change_nodes, delta, abs_delta, scalar_features),
                dim=-1,
            )
            residual_update = self.residual_encoder(residual_input)
            gate_input = torch.cat(
                (
                    support_nodes,
                    deny_nodes,
                    change_nodes,
                    abs_delta,
                    scalar_features,
                ),
                dim=-1,
            )
            node_gate = torch.sigmoid(
                self.residual_gate(gate_input).squeeze(-1)
            )
            node_gate = (
                self.residual_gate_floor
                + (1.0 - self.residual_gate_floor) * node_gate
            )
            conflict_nodes = self.residual_norm(
                change_nodes
                + self.residual_scale
                * node_gate.unsqueeze(-1)
                * residual_update
            )
            readout_weight = (
                (1.0 - self.readout_strength)
                + self.readout_strength * conflict_score
            )
            readout_weight = readout_weight * node_gate
            if base_node_keep is not None:
                readout_weight = readout_weight * base_node_keep.clamp(
                    0.0,
                    1.0,
                )
            if (
                self.force_root_keep
                and roots is not None
                and roots.numel() > 0
            ):
                readout_weight = readout_weight.clone()
                readout_weight[roots] = 1.0
            keep_sample = readout_weight.clamp(0.0, 1.0)
            graph_hidden = self._pool_nodes(
                conflict_nodes,
                keep_sample,
                batch,
                pool_is_sum,
            )
        else:
            keep_sample = self.soft_bernoulli_sample(keep_probability)
            if (
                self.force_root_keep
                and roots is not None
                and roots.numel() > 0
            ):
                keep_sample = keep_sample.clone()
                keep_sample[roots] = 1.0

            scalar_features = torch.stack(
                (field, conflict_intensity, high_frequency),
                dim=-1,
            )
            node_input = torch.cat(
                (change_nodes, delta, abs_delta, scalar_features),
                dim=-1,
            )
            conflict_nodes = self.node_encoder(node_input)
            graph_hidden = self._pool_nodes(
                conflict_nodes,
                keep_sample,
                batch,
                pool_is_sum,
            )

        zero = graph_hidden.new_zeros(())
        label_loss = zero
        if target is not None and self.lambda_label > 0.0:
            label_logits = self.label_head(graph_hidden)
            label_loss = self.lambda_label * F.cross_entropy(
                label_logits,
                target,
            )

        roots_mask = self._regularizer_mask(
            keep_probability.size(0),
            roots,
            keep_probability.device,
            keep_probability.dtype,
        )
        size_loss = zero
        if self.lambda_size > 0.0:
            regularized_probability = (
                conflict_score
                if self.encoder_mode == "transformer"
                else keep_sample
            )
            size_loss = self.lambda_size * self._masked_mean(
                regularized_probability,
                roots_mask,
            )

        redundancy_loss = zero
        if (
            original_graph is not None
            and self.lambda_redundancy > 0.0
            and original_graph.numel() > 0
        ):
            conflict_norm = F.normalize(
                graph_hidden,
                p=2,
                dim=-1,
                eps=self.eps,
            )
            original_norm = F.normalize(
                original_graph.detach(),
                p=2,
                dim=-1,
                eps=self.eps,
            )
            redundancy_loss = (
                self.lambda_redundancy
                * (conflict_norm * original_norm).sum(dim=-1).pow(2).mean()
            )

        aux_loss = label_loss + size_loss + redundancy_loss
        outputs = {
            "field": field,
            "support_strength": support_strength,
            "deny_strength": deny_strength,
            "balance_conflict": balance_conflict,
            "evidence_energy": evidence_energy,
            "conflict_intensity": conflict_intensity,
            "high_frequency": high_frequency,
            "conflict_score": conflict_score,
            "keep_probability": keep_probability,
            "keep_sample": keep_sample,
            "attention_score": conflict_score,
            "attention_received": attention_received,
            "attention_by_head": attention_by_head,
            "node_importance": node_importance,
            "node_gate": node_gate,
            "encoder_mode": self.encoder_mode,
            "nodes": conflict_nodes,
            "graph": graph_hidden,
            "aux_loss": aux_loss,
            "label_loss": label_loss,
            "size_loss": size_loss,
            "redundancy_loss": redundancy_loss,
        }
        return graph_hidden, conflict_nodes, outputs


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
      -> optional collective reinforcement/revision-response branch
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
        self.use_vertical_path_attention = bool(
            getattr(args, "use_vertical_path_attention", False)
        )
        self.use_semantic_tree_transformer = bool(
            getattr(args, "use_semantic_tree_transformer", False)
        )
        self.use_node_keep_in_change_pool = bool(
            getattr(args, "use_node_keep_in_change_pool", True)
        )
        self.use_conflict_field_bottleneck = bool(
            getattr(args, "use_conflict_field_bottleneck", False)
        )
        requested_fusion_mode = getattr(
            args,
            "classification_fusion_mode",
            None,
        )
        if requested_fusion_mode is None:
            requested_branches = ["original", "change"]
            if self.use_trend_graph:
                requested_branches.append("trend")
            if self.use_vertical_path_attention:
                requested_branches.append("vertical")
            if self.use_semantic_tree_transformer:
                requested_branches.append("semantic_tree")
            if self.use_conflict_field_bottleneck:
                requested_branches.append("conflict")
            requested_fusion_mode = "_".join(requested_branches)
        self.classification_fusion_mode = str(
            requested_fusion_mode
        ).strip().lower()
        fusion_mode_branches = {
            "change": ("change",),
            "semantic_tree": ("semantic_tree",),
            "conflict": ("conflict",),
            "change_conflict": ("change", "conflict"),
            "collective_revision": ("collective_revision",),
            "change_collective_revision": (
                "change",
                "collective_revision",
            ),
            "change_conflict_collective_revision": (
                "change",
                "conflict",
                "collective_revision",
            ),
            "original_change": ("original", "change"),
            "original_change_conflict": (
                "original",
                "change",
                "conflict",
            ),
            "original_change_collective_revision": (
                "original",
                "change",
                "collective_revision",
            ),
            "original_change_conflict_collective_revision": (
                "original",
                "change",
                "conflict",
                "collective_revision",
            ),
            "original_change_trend": (
                "original",
                "change",
                "trend",
            ),
            "original_change_trend_conflict": (
                "original",
                "change",
                "trend",
                "conflict",
            ),
            "original_change_trend_collective_revision": (
                "original",
                "change",
                "trend",
                "collective_revision",
            ),
            "original_change_trend_conflict_collective_revision": (
                "original",
                "change",
                "trend",
                "conflict",
                "collective_revision",
            ),
            "original_change_vertical": (
                "original",
                "change",
                "vertical",
            ),
            "original_change_vertical_conflict": (
                "original",
                "change",
                "vertical",
                "conflict",
            ),
            "original_change_trend_vertical": (
                "original",
                "change",
                "trend",
                "vertical",
            ),
            "original_change_trend_vertical_conflict": (
                "original",
                "change",
                "trend",
                "vertical",
                "conflict",
            ),
            "support_deny_conflict": (
                "support",
                "deny",
                "conflict",
            ),
            "support_deny_change": (
                "support",
                "deny",
                "change",
            ),
            "support_deny_change_conflict": (
                "support",
                "deny",
                "change",
                "conflict",
            ),
            "support_deny_change_collective_revision": (
                "support",
                "deny",
                "change",
                "collective_revision",
            ),
            "support_deny_change_conflict_collective_revision": (
                "support",
                "deny",
                "change",
                "conflict",
                "collective_revision",
            ),
            "support_deny_change_vertical": (
                "support",
                "deny",
                "change",
                "vertical",
            ),
            "support_deny_change_vertical_conflict": (
                "support",
                "deny",
                "change",
                "vertical",
                "conflict",
            ),
        }
        for mode_name, branch_names in list(fusion_mode_branches.items()):
            if "semantic_tree" in branch_names:
                continue
            fusion_mode_branches[mode_name + "_semantic_tree"] = (
                branch_names + ("semantic_tree",)
            )
        fusion_mode_branches.update(
            {
                "change_semantic_tree_conflict": (
                    "change",
                    "semantic_tree",
                    "conflict",
                ),
                "original_change_semantic_tree_conflict": (
                    "original",
                    "change",
                    "semantic_tree",
                    "conflict",
                ),
                "support_deny_change_semantic_tree_conflict": (
                    "support",
                    "deny",
                    "change",
                    "semantic_tree",
                    "conflict",
                ),
            }
        )
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
        self.vertical_path_active = (
            self.use_vertical_path_attention
            or "vertical" in self.classification_branch_names
        )
        self.semantic_tree_active = (
            self.use_semantic_tree_transformer
            or "semantic_tree" in self.classification_branch_names
        )
        self.conflict_field_active = (
            self.use_conflict_field_bottleneck
            or "conflict" in self.classification_branch_names
        )
        self.collective_revision_active = (
            "collective_revision" in self.classification_branch_names
        )
        self.use_global_ds_fusion = bool(
            getattr(args, "use_global_ds_fusion", False)
        )
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
        self.lambda_view_mi = max(
            0.0,
            float(getattr(args, "lambda_view_mi_aux", 0.0)),
        )
        self.lambda_ds_unknown_edge = max(
            0.0,
            float(getattr(args, "lambda_ds_unknown_edge_aux", 0.0)),
        )
        self.lambda_semantic_change_bottleneck = max(
            0.0,
            float(getattr(args, "lambda_semantic_change_bottleneck", 0.0)),
        )
        self.view_mi_eps = max(
            1e-12,
            float(getattr(args, "view_mi_eps", 1e-6)),
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
        self.classification_head_mode = str(
            getattr(args, "classification_head_mode", "fusion")
        ).strip().lower()
        valid_head_modes = {"fusion", "branch_sum"}
        if self.classification_head_mode not in valid_head_modes:
            raise ValueError(
                "classification_head_mode must be one of {}, got {}".format(
                    sorted(valid_head_modes),
                    self.classification_head_mode,
                )
            )
        self.register_buffer(
            "classification_branch_weights",
            self._classification_branch_weight_tensor(
                getattr(args, "classification_branch_weights", None),
            ),
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
        self.use_semantic_parity_gnn = bool(
            getattr(args, "use_semantic_parity_gnn", True)
        )
        self.semantic_node_weight_mode = str(
            getattr(args, "semantic_node_weight_mode", "local")
        ).strip().lower()
        valid_node_weight_modes = {"local", "root_parity", "parity"}
        if self.semantic_node_weight_mode not in valid_node_weight_modes:
            raise ValueError(
                "semantic_node_weight_mode must be one of {}, got {}".format(
                    sorted(valid_node_weight_modes),
                    self.semantic_node_weight_mode,
                )
            )
        parity_layers = max(1, int(getattr(args, "n_layers_conv", 2)))
        self.semantic_parity_encoder = (
            SemanticParityEncoder(
                input_dim=in_feats,
                hidden_dim=hid_feats,
                num_layers=parity_layers,
                dropout=self.dropout,
                bidirectional=self.backbone_type == "bigcn",
                residual=bool(
                    getattr(args, "semantic_parity_residual", True)
                ),
            )
            if self.use_semantic_parity_gnn
            else None
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
            args=args,
        )
        self.vertical_path_attention = (
            RootPathUncertaintyAttention(
                hid_feats,
                args=args,
            )
            if self.vertical_path_active
            else None
        )
        self.semantic_tree_transformer = (
            SemanticTreeTransformerBranch(
                hid_feats,
                args=args,
            )
            if self.semantic_tree_active
            else None
        )
        self.conflict_field_bottleneck = (
            ConflictFieldBottleneck(
                hid_feats,
                num_classes,
                args=args,
            )
            if self.conflict_field_active
            else None
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
        self.collective_revision_encoder = (
            CollectiveRevisionEncoder(hid_feats, args)
            if self.collective_revision_active
            else None
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
        self.branch_classifiers = nn.ModuleDict()
        if self.classification_head_mode == "branch_sum":
            for branch_name in self.classification_branch_names:
                self.branch_classifiers[branch_name] = nn.Linear(
                    hid_feats,
                    num_classes,
                    bias=False,
                )
            self.branch_sum_bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("branch_sum_bias", None)
        self.global_ds_fusion = (
            GlobalDSFusionHead(
                hid_feats,
                num_classes,
                self.classification_branch_names,
                args=args,
            )
            if self.use_global_ds_fusion
            else None
        )

        self._last_aux_loss = None
        self._last_edge_relation_loss = None
        self._last_view_mi_loss = None
        self._last_semantic_change_bottleneck_loss = None
        self._last_branch_logits = None
        self._last_global_ds_masses = None
        self._last_global_ds_branch_masses = None
        self._last_global_ds_conflict = None
        self._last_edge_probabilities = None
        self._last_edge_dirichlet_alpha = None
        self._last_edge_masses = None
        self._last_edge_unknown_mass = None
        self._last_edge_uncertainty = None
        self._last_keep_sample = None
        self._last_support_weight = None
        self._last_deny_weight = None
        self._last_original_nodes = None
        self._last_original_graph = None
        self._last_support_graph = None
        self._last_deny_graph = None
        self._last_change_nodes = None
        self._last_change_graph = None
        self._last_vertical_nodes = None
        self._last_vertical_graph = None
        self._last_semantic_tree_nodes = None
        self._last_semantic_tree_graph = None
        self._last_semantic_tree_depth = None
        self._last_semantic_tree_topics = None
        self._last_semantic_tree_topic_similarity = None
        self._last_semantic_tree_attention = None
        self._last_node_uncertainty = None
        self._last_trend_sequence = None
        self._last_node_state_sequence = None
        self._last_collective_revision_graph = None
        self._last_collective_revision_outputs = None
        self._last_node_keep = None
        self._last_child_degree_importance = None
        self._last_conflict_field = None
        self._last_conflict_support_strength = None
        self._last_conflict_deny_strength = None
        self._last_conflict_balance = None
        self._last_conflict_energy = None
        self._last_conflict_intensity = None
        self._last_conflict_high_frequency = None
        self._last_conflict_score = None
        self._last_conflict_keep_probability = None
        self._last_conflict_keep_sample = None
        self._last_conflict_attention_score = None
        self._last_conflict_attention_received = None
        self._last_conflict_attention_by_head = None
        self._last_conflict_node_importance = None
        self._last_conflict_node_gate = None
        self._last_conflict_encoder_mode = None
        self._last_conflict_nodes = None
        self._last_conflict_graph = None
        self._last_conflict_aux_loss = None
        self._last_conflict_label_loss = None
        self._last_conflict_size_loss = None
        self._last_conflict_redundancy_loss = None

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
        self.bigcn_num_layers = max(
            2,
            int(getattr(args, "n_layers_conv", 2)),
        )
        self.td_conv1 = GCNConv(in_feats, hid_feats)
        self.td_conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.bu_conv1 = GCNConv(in_feats, hid_feats)
        self.bu_conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.td_extra_convs = nn.ModuleList(
            [
                GCNConv(out_feats + out_feats, out_feats)
                for _ in range(self.bigcn_num_layers - 2)
            ]
        )
        self.bu_extra_convs = nn.ModuleList(
            [
                GCNConv(out_feats + out_feats, out_feats)
                for _ in range(self.bigcn_num_layers - 2)
            ]
        )
        branch_dim = out_feats + hid_feats
        self.direction_fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, hid_feats),
            nn.ReLU(),
        )

    def set_epoch(self, epoch):
        self.edge_router.set_epoch(epoch)
        if self.conflict_field_bottleneck is not None:
            self.conflict_field_bottleneck.set_epoch(epoch)

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

    def _classification_branch_weight_tensor(self, raw_weights):
        weights = [1.0] * len(self.classification_branch_names)
        if raw_weights is None:
            return torch.tensor(weights, dtype=torch.float32)

        if isinstance(raw_weights, str):
            raw_weights = raw_weights.strip()
            if raw_weights == "":
                return torch.tensor(weights, dtype=torch.float32)
            parts = [
                part.strip()
                for part in raw_weights.split(",")
                if part.strip()
            ]
            if any(("=" in part or ":" in part) for part in parts):
                parsed_weights = {}
                for part in parts:
                    if "=" in part:
                        branch_name, value = part.split("=", 1)
                    elif ":" in part:
                        branch_name, value = part.split(":", 1)
                    else:
                        raise ValueError(
                            "classification_branch_weights mixes named "
                            "and positional values: {}".format(raw_weights)
                        )
                    parsed_weights[branch_name.strip()] = float(
                        value.strip()
                    )
                raw_weights = parsed_weights
            else:
                raw_weights = [float(part) for part in parts]

        if isinstance(raw_weights, dict):
            branch_to_index = {
                branch_name: index
                for index, branch_name in enumerate(
                    self.classification_branch_names
                )
            }
            for raw_name, value in raw_weights.items():
                branch_name = str(raw_name).strip().lower()
                if branch_name not in branch_to_index:
                    raise ValueError(
                        "classification_branch_weights contains unknown "
                        "branch '{}'; expected one of {}".format(
                            raw_name,
                            list(self.classification_branch_names),
                        )
                    )
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(
                        "classification_branch_weights for branch '{}' "
                        "must be finite".format(raw_name)
                    )
                weights[branch_to_index[branch_name]] = value
        else:
            if torch.is_tensor(raw_weights):
                values = raw_weights.detach().cpu().view(-1).tolist()
            else:
                try:
                    values = list(raw_weights)
                except TypeError as exc:
                    raise ValueError(
                        "classification_branch_weights must be a dict, "
                        "comma-separated string, or sequence"
                    ) from exc
            if len(values) != len(self.classification_branch_names):
                raise ValueError(
                    "classification_branch_weights must provide {} values "
                    "for branches {}, got {}".format(
                        len(self.classification_branch_names),
                        list(self.classification_branch_names),
                        len(values),
                    )
                )
            weights = [float(value) for value in values]
            if not all(math.isfinite(value) for value in weights):
                raise ValueError(
                    "classification_branch_weights values must be finite"
                )

        return torch.tensor(weights, dtype=torch.float32)

    def _branch_sum_logits(self, classification_graphs):
        branch_weights = self.classification_branch_weights.to(
            device=classification_graphs[0].device,
            dtype=classification_graphs[0].dtype,
        )
        logits = None
        branch_logits = {}
        for index, branch_name in enumerate(self.classification_branch_names):
            weighted_graph = classification_graphs[index] * branch_weights[
                index
            ]
            branch_logit = self.branch_classifiers[branch_name](
                weighted_graph
            )
            branch_logits[branch_name] = branch_logit
            logits = branch_logit if logits is None else logits + branch_logit
        if self.branch_sum_bias is not None:
            logits = logits + self.branch_sum_bias
        self._last_branch_logits = {
            name: value.detach()
            for name, value in branch_logits.items()
        }
        return logits

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
        extra_convs=None,
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
        for conv in extra_convs or []:
            root_current = self._extend_root_features(hidden, data)
            hidden = torch.cat((hidden, root_current), dim=-1)
            hidden = F.relu(hidden)
            hidden = F.dropout(
                hidden,
                p=self.dropout,
                training=self.training,
            )
            hidden = conv(hidden, edge_index, edge_weight=edge_weight)
            hidden = F.relu(hidden)
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
            self.td_extra_convs,
        )
        bottom_up = self._encode_bigcn_direction(
            data,
            self._reverse_edges(edge_index),
            edge_weight,
            self.bu_conv1,
            self.bu_conv2,
            self.bu_extra_convs,
        )
        return self.direction_fusion(
            torch.cat((top_down, bottom_up), dim=-1)
        )

    def _encode_semantic_views(
        self,
        data,
        node_hidden,
        edge_index,
        support_weight,
        deny_weight,
    ):
        if self.semantic_parity_encoder is not None:
            return self.semantic_parity_encoder(
                data.x.float(),
                edge_index,
                support_weight,
                deny_weight,
            )
        support_nodes = self._encode_semantic_view(
            data,
            node_hidden,
            edge_index,
            support_weight,
        )
        deny_nodes = self._encode_semantic_view(
            data,
            node_hidden,
            edge_index,
            deny_weight,
        )
        return support_nodes, deny_nodes

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
        if self.edge_router.use_ds_mass_routing:
            masses, _, _ = self.edge_router.relation_masses(logits[valid])
            pignistic = torch.stack(
                (
                    masses[:, 0] + 0.5 * masses[:, 1],
                    masses[:, 2] + 0.5 * masses[:, 1],
                ),
                dim=-1,
            ).clamp_min(self.view_mi_eps)
            edge_loss = F.nll_loss(
                pignistic.log(),
                valid_labels,
                weight=class_weight,
            )
            if self.lambda_ds_unknown_edge > 0.0:
                edge_loss = (
                    edge_loss
                    + self.lambda_ds_unknown_edge * masses[:, 1].mean()
                )
            return relation_weight * edge_loss
        if self.edge_router.use_dirichlet_relation_routing:
            concentration = self.edge_router.dirichlet_concentration(
                logits[valid]
            ).clamp_min(self.view_mi_eps)
            teacher = self.edge_router.dirichlet_teacher_concentration(
                valid_labels
            ).to(device=concentration.device, dtype=concentration.dtype)
            teacher = teacher.clamp_min(self.view_mi_eps)
            edge_kl = kl_divergence(
                Dirichlet(teacher),
                Dirichlet(concentration),
            )
            if class_weight is not None:
                sample_weight = class_weight[valid_labels].to(
                    device=edge_kl.device,
                    dtype=edge_kl.dtype,
                )
                edge_loss = (
                    edge_kl * sample_weight
                ).sum() / sample_weight.sum().clamp_min(self.view_mi_eps)
            else:
                edge_loss = edge_kl.mean()
            return relation_weight * edge_loss
        return relation_weight * F.cross_entropy(
            logits[valid],
            valid_labels,
            weight=class_weight,
        )

    def _view_mutual_information_loss(
        self,
        support_graph,
        deny_graph,
    ):
        zero = self.classifier.weight.new_zeros(())
        if self.lambda_view_mi <= 0.0:
            return zero
        if support_graph is None or deny_graph is None:
            return zero
        if support_graph.numel() == 0 or deny_graph.numel() == 0:
            return zero

        support = support_graph.float()
        deny = deny_graph.float()
        if support.size(0) > 1:
            support = support - support.mean(dim=0, keepdim=True)
            deny = deny - deny.mean(dim=0, keepdim=True)
            support = support / support.pow(2).mean(
                dim=0,
                keepdim=True,
            ).add(self.view_mi_eps).sqrt()
            deny = deny / deny.pow(2).mean(
                dim=0,
                keepdim=True,
            ).add(self.view_mi_eps).sqrt()
            cross_covariance = support.t().matmul(deny) / support.size(0)
            mi_proxy = cross_covariance.pow(2).mean()
        else:
            support = F.normalize(
                support,
                p=2,
                dim=-1,
                eps=self.view_mi_eps,
            )
            deny = F.normalize(
                deny,
                p=2,
                dim=-1,
                eps=self.view_mi_eps,
            )
            mi_proxy = (support * deny).sum(dim=-1).pow(2).mean()
        return self.lambda_view_mi * mi_proxy

    def _semantic_change_bottleneck_loss(self):
        zero = self.classifier.weight.new_zeros(())
        if self.lambda_semantic_change_bottleneck <= 0.0:
            return zero
        kl_loss = getattr(self.semantic_change_encoder, "kl_loss", None)
        if kl_loss is None:
            return zero
        return self.lambda_semantic_change_bottleneck * kl_loss()

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

    def _build_parity_view_node_weights(
        self,
        data,
        support_weight,
        deny_weight,
    ):
        support_node = support_weight.new_zeros(data.x.size(0))
        deny_node = support_weight.new_zeros(data.x.size(0))
        roots = self._root_indices(data)
        support_node[roots] = 1.0
        if data.edge_index.numel() == 0:
            # Keep the source post as a shared anchor when a view is pooled or
            # passed to the semantic tree branch.
            deny_node[roots] = 1.0
            return support_node, deny_node

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
            edge_support = support_weight[edge_ids].clamp(0.0, 1.0)
            edge_deny = deny_weight[edge_ids].clamp(0.0, 1.0)
            parent_support = support_node[parent]
            parent_deny = deny_node[parent]
            support_node[child] = (
                parent_support * edge_support
                + parent_deny * edge_deny
            )
            deny_node[child] = (
                parent_support * edge_deny
                + parent_deny * edge_support
            )

        # The root has no stance relative to itself, but downstream modules use
        # it as an aligned source anchor for both views.
        deny_node[roots] = 1.0
        return support_node.clamp(0.0, 1.0), deny_node.clamp(0.0, 1.0)

    def _build_semantic_node_weights(
        self,
        data,
        support_weight,
        deny_weight,
    ):
        if self.semantic_node_weight_mode in {"root_parity", "parity"}:
            return self._build_parity_view_node_weights(
                data,
                support_weight,
                deny_weight,
            )
        return (
            self._build_view_node_weight(data, support_weight),
            self._build_view_node_weight(data, deny_weight),
        )

    def _build_path_parent_and_uncertainty(
        self,
        data,
        edge_uncertainty,
    ):
        num_nodes = data.x.size(0)
        parent = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=data.x.device,
        )
        node_uncertainty = data.x.new_zeros(num_nodes)
        depth = self._node_depths(data, data.edge_index)
        roots = self._root_indices(data)
        parent[roots] = -1
        node_uncertainty[roots] = 0.0
        if data.edge_index.numel() == 0 or edge_uncertainty.numel() == 0:
            return parent, depth, node_uncertainty

        src, dst = data.edge_index
        valid_edge = (
            (depth[src] >= 0)
            & (depth[dst] == depth[src] + 1)
        )
        edge_ids = valid_edge.nonzero(as_tuple=False).view(-1)
        if edge_ids.numel() == 0:
            return parent, depth, node_uncertainty

        child = dst[edge_ids]
        parent[child] = src[edge_ids]
        node_uncertainty[child] = edge_uncertainty[edge_ids]
        parent[roots] = -1
        node_uncertainty[roots] = 0.0
        return parent, depth, node_uncertainty

    def _encode_original_graph_nodes(self, data, node_hidden):
        # The original-graph branch keeps every propagation edge. It shares the
        # dataset-specific BiGCN/ResGCN backbone with the semantic views, but is
        # independent of uncertainty sampling and support/deny routing.
        original_edge_weight = data.x.new_ones(data.edge_index.size(1))
        return self._encode_semantic_view(
            data,
            node_hidden,
            data.edge_index,
            original_edge_weight,
        )

    def _encode_original_graph(self, data, node_hidden):
        original_nodes = self._encode_original_graph_nodes(data, node_hidden)
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
        if hasattr(self.semantic_change_encoder, "last_kl_loss"):
            self.semantic_change_encoder.last_kl_loss = None
            self.semantic_change_encoder.last_mean = None
            self.semantic_change_encoder.last_logvar = None

        node_hidden = self.node_projection(data.x.float())
        node_hidden = self._add_root_context(node_hidden, data)
        original_nodes = None
        original_graph = None
        needs_original_nodes = (
            "original" in self.classification_branch_names
            or self.semantic_tree_transformer is not None
        )
        if needs_original_nodes:
            original_nodes = self._encode_original_graph_nodes(
                data,
                node_hidden,
            )
        if "original" in self.classification_branch_names:
            original_graph = self.global_pool(original_nodes, data.batch)
        child_degree_importance = self._child_degree_importance(data)

        (
            relation_logits,
            probabilities,
            edge_uncertainty,
        ) = self.edge_router.relation_outputs(
            node_hidden,
            data.edge_index,
        )


#### vertical path 模块，现在不开启
        vertical_nodes = None
        vertical_graph = None
        node_uncertainty = None
        path_depth = None
        if self.vertical_path_attention is not None:
            (
                path_parent,
                path_depth,
                node_uncertainty,
            ) = self._build_path_parent_and_uncertainty(
                data,
                edge_uncertainty,
            )
            vertical_nodes = self.vertical_path_attention(
                node_hidden,
                path_parent,
                path_depth,
                node_uncertainty,
            )
            vertical_graph = self.global_pool(vertical_nodes, data.batch)
        (
            keep_sample,
            support_weight,
            deny_weight,
        ) = self.edge_router.route_edges(
            relation_logits,
            probabilities,
            edge_uncertainty,
            child_degree_importance,
        )

        #这里现在不使用伯努利采样，默认设置node_keep为1
        node_keep = self._build_root_connected_keep(
            data,
            keep_sample,
        )
        if data.edge_index.numel() > 0:
            parent_keep = node_keep[data.edge_index[0]]
            support_weight = support_weight * parent_keep
            deny_weight = deny_weight * parent_keep

        support_nodes, deny_nodes = self._encode_semantic_views(
            data,
            node_hidden,
            data.edge_index,
            support_weight,
            deny_weight,
        )
        (
            support_node_weight,
            deny_node_weight,
        ) = self._build_semantic_node_weights(
            data,
            support_weight,
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
        conflict_needs_change = (
            self.conflict_field_bottleneck is not None
            and self.conflict_field_bottleneck.encoder_mode
            in {"pool", "residual"}
        )
        needs_change_nodes = (
            "change" in self.classification_branch_names
            or conflict_needs_change
        )
        change_nodes = None
        change_graph = None
        if needs_change_nodes:
            change_nodes = self.semantic_change_encoder(
                support_nodes,
                deny_nodes,
                batch=data.batch,
                edge_index=data.edge_index,
                support_node_weight=support_node_weight,
                deny_node_weight=deny_node_weight,
                node_keep=node_keep,
            )
            if self.use_node_keep_in_change_pool:
                change_graph = self._pool_root_connected_nodes(
                    change_nodes,
                    node_keep,
                    data.batch,
                )
            else:
                change_graph = self.global_pool(change_nodes, data.batch)



        semantic_tree_graph = None
        semantic_tree_nodes = None
        semantic_tree_depth = None
        if self.semantic_tree_transformer is not None:
            semantic_tree_depth = (
                path_depth
                if path_depth is not None
                else self._node_depths(data, data.edge_index)
            )
            (
                semantic_tree_graph,
                semantic_tree_nodes,
            ) = self.semantic_tree_transformer(
                original_nodes,
                support_nodes,
                deny_nodes,
                semantic_tree_depth,
                data.batch,
                support_node_weight=support_node_weight,
                deny_node_weight=deny_node_weight,
            )

        conflict_graph = None
        conflict_nodes = None
        conflict_outputs = None
        if self.conflict_field_bottleneck is not None:
            conflict_depth = semantic_tree_depth
            if (
                self.conflict_field_bottleneck.encoder_mode == "transformer"
                and conflict_depth is None
            ):
                conflict_depth = self._node_depths(data, data.edge_index)
            original_context = (
                original_graph
                if original_graph is not None
                else self.global_pool(node_hidden, data.batch)
            )
            (
                conflict_graph,
                conflict_nodes,
                conflict_outputs,
            ) = self.conflict_field_bottleneck(
                support_nodes,
                deny_nodes,
                change_nodes,
                data.edge_index,
                data.batch,
                self._root_indices(data),
                pool_is_sum=self.pool_is_sum,
                original_graph=original_context,
                target=getattr(data, "y", None),
                base_node_keep=node_keep,
                original_nodes=node_hidden,
                depth=conflict_depth,
            )

        trend_sequence = self._build_uncertainty_trend(
            data,
            probabilities,
            keep_sample,
        )
        trend_graph = None
        if "trend" in self.classification_branch_names:
            trend_graph = self._encode_trend(
                trend_sequence,
                data.num_hop,
            )
        collective_revision_graph = None
        collective_revision_outputs = None
        if self.collective_revision_encoder is not None:
            (
                collective_revision_graph,
                collective_revision_outputs,
            ) = self.collective_revision_encoder(
                trend_sequence,
                data.num_hop,
            )

        graph_branches = {
            "original": original_graph,
            "support": support_graph,
            "deny": deny_graph,
            "change": change_graph,
            "trend": trend_graph,
            "collective_revision": collective_revision_graph,
            "vertical": vertical_graph,
            "semantic_tree": semantic_tree_graph,
            "conflict": conflict_graph,
        }
        classification_graphs = [
            graph_branches[name]
            for name in self.classification_branch_names
        ]
        if self.global_ds_fusion is not None:
            (
                output_log_prob,
                global_ds_masses,
                global_ds_branch_masses,
                global_ds_conflict,
            ) = self.global_ds_fusion(graph_branches)
        else:
            if self.classification_head_mode == "branch_sum":
                logits = self._branch_sum_logits(classification_graphs)
            else:
                self._last_branch_logits = None
                fused = self.fusion(
                    torch.cat(classification_graphs, dim=-1)
                )
                logits = self.classifier(fused)
            output_log_prob = F.log_softmax(logits, dim=-1)
            global_ds_masses = None
            global_ds_branch_masses = None
            global_ds_conflict = None

        edge_relation_loss = self._edge_relation_loss(
            relation_logits,
            getattr(data, "edge_stance", None),
        )
        view_mi_loss = self._view_mutual_information_loss(
            support_graph,
            deny_graph,
        )
        change_bottleneck_loss = self._semantic_change_bottleneck_loss()
        conflict_aux_loss = (
            relation_logits.new_zeros(())
            if conflict_outputs is None
            else conflict_outputs["aux_loss"]
        )
        self._last_aux_loss = (
            edge_relation_loss
            + view_mi_loss
            + change_bottleneck_loss
            + conflict_aux_loss
        )
        self._last_edge_relation_loss = edge_relation_loss.detach()
        self._last_view_mi_loss = view_mi_loss.detach()
        self._last_semantic_change_bottleneck_loss = (
            change_bottleneck_loss.detach()
        )
        self._last_conflict_aux_loss = conflict_aux_loss.detach()
        self._last_global_ds_masses = (
            None
            if global_ds_masses is None
            else global_ds_masses.detach()
        )
        self._last_global_ds_branch_masses = (
            None
            if global_ds_branch_masses is None
            else global_ds_branch_masses.detach()
        )
        self._last_global_ds_conflict = (
            None
            if global_ds_conflict is None
            else global_ds_conflict.detach()
        )
        self._last_edge_probabilities = probabilities.detach()
        if self.edge_router.use_dirichlet_relation_routing:
            self._last_edge_dirichlet_alpha = (
                self.edge_router.dirichlet_concentration(relation_logits)
                .detach()
            )
            self._last_edge_masses = None
            self._last_edge_unknown_mass = None
        elif self.edge_router.use_ds_mass_routing:
            self._last_edge_dirichlet_alpha = None
            edge_masses, _, edge_unknown_mass = (
                self.edge_router.relation_masses(relation_logits)
            )
            self._last_edge_masses = edge_masses.detach()
            self._last_edge_unknown_mass = edge_unknown_mass.detach()
        else:
            self._last_edge_dirichlet_alpha = None
            self._last_edge_masses = None
            self._last_edge_unknown_mass = None
        self._last_edge_uncertainty = edge_uncertainty.detach()
        self._last_keep_sample = keep_sample.detach()
        self._last_support_weight = support_weight.detach()
        self._last_deny_weight = deny_weight.detach()
        self._last_original_nodes = (
            None if original_nodes is None else original_nodes.detach()
        )
        self._last_original_graph = (
            None if original_graph is None else original_graph.detach()
        )
        self._last_support_graph = support_graph.detach()
        self._last_deny_graph = deny_graph.detach()
        self._last_change_nodes = (
            None if change_nodes is None else change_nodes.detach()
        )
        self._last_change_graph = (
            None if change_graph is None else change_graph.detach()
        )
        self._last_vertical_nodes = (
            None if vertical_nodes is None else vertical_nodes.detach()
        )
        self._last_vertical_graph = (
            None if vertical_graph is None else vertical_graph.detach()
        )
        self._last_semantic_tree_nodes = (
            None
            if semantic_tree_nodes is None
            else semantic_tree_nodes.detach()
        )
        self._last_semantic_tree_graph = (
            None
            if semantic_tree_graph is None
            else semantic_tree_graph.detach()
        )
        self._last_semantic_tree_depth = (
            None
            if semantic_tree_depth is None
            else semantic_tree_depth.detach()
        )
        self._last_semantic_tree_topics = (
            None
            if self.semantic_tree_transformer is None
            else self.semantic_tree_transformer.last_topic_distribution
            .detach()
        )
        self._last_semantic_tree_topic_similarity = (
            None
            if self.semantic_tree_transformer is None
            else self.semantic_tree_transformer.last_topic_similarity.detach()
        )
        self._last_semantic_tree_attention = (
            None
            if self.semantic_tree_transformer is None
            else self.semantic_tree_transformer.last_attention.detach()
        )
        if conflict_outputs is None:
            self._last_conflict_field = None
            self._last_conflict_support_strength = None
            self._last_conflict_deny_strength = None
            self._last_conflict_balance = None
            self._last_conflict_energy = None
            self._last_conflict_intensity = None
            self._last_conflict_high_frequency = None
            self._last_conflict_score = None
            self._last_conflict_keep_probability = None
            self._last_conflict_keep_sample = None
            self._last_conflict_attention_score = None
            self._last_conflict_attention_received = None
            self._last_conflict_attention_by_head = None
            self._last_conflict_node_importance = None
            self._last_conflict_node_gate = None
            self._last_conflict_encoder_mode = None
            self._last_conflict_nodes = None
            self._last_conflict_graph = None
            self._last_conflict_label_loss = None
            self._last_conflict_size_loss = None
            self._last_conflict_redundancy_loss = None
        else:
            self._last_conflict_field = conflict_outputs["field"].detach()
            self._last_conflict_support_strength = (
                conflict_outputs["support_strength"].detach()
            )
            self._last_conflict_deny_strength = (
                conflict_outputs["deny_strength"].detach()
            )
            self._last_conflict_balance = (
                conflict_outputs["balance_conflict"].detach()
            )
            self._last_conflict_energy = (
                conflict_outputs["evidence_energy"].detach()
            )
            self._last_conflict_intensity = (
                conflict_outputs["conflict_intensity"].detach()
            )
            self._last_conflict_high_frequency = (
                conflict_outputs["high_frequency"].detach()
            )
            self._last_conflict_score = (
                conflict_outputs["conflict_score"].detach()
            )
            self._last_conflict_keep_probability = (
                conflict_outputs["keep_probability"].detach()
            )
            self._last_conflict_keep_sample = (
                conflict_outputs["keep_sample"].detach()
            )
            self._last_conflict_attention_score = (
                conflict_outputs["attention_score"].detach()
            )
            self._last_conflict_attention_received = (
                None
                if conflict_outputs["attention_received"] is None
                else conflict_outputs["attention_received"].detach()
            )
            self._last_conflict_attention_by_head = (
                None
                if conflict_outputs["attention_by_head"] is None
                else conflict_outputs["attention_by_head"].detach()
            )
            self._last_conflict_node_importance = (
                None
                if conflict_outputs["node_importance"] is None
                else conflict_outputs["node_importance"].detach()
            )
            self._last_conflict_node_gate = (
                None
                if conflict_outputs["node_gate"] is None
                else conflict_outputs["node_gate"].detach()
            )
            self._last_conflict_encoder_mode = conflict_outputs[
                "encoder_mode"
            ]
            self._last_conflict_nodes = conflict_outputs["nodes"].detach()
            self._last_conflict_graph = conflict_outputs["graph"].detach()
            self._last_conflict_label_loss = (
                conflict_outputs["label_loss"].detach()
            )
            self._last_conflict_size_loss = (
                conflict_outputs["size_loss"].detach()
            )
            self._last_conflict_redundancy_loss = (
                conflict_outputs["redundancy_loss"].detach()
            )
        self._last_node_uncertainty = (
            None if node_uncertainty is None else node_uncertainty.detach()
        )
        self._last_trend_sequence = trend_sequence.detach()
        self._last_collective_revision_graph = (
            None
            if collective_revision_graph is None
            else collective_revision_graph.detach()
        )
        self._last_collective_revision_outputs = (
            None
            if collective_revision_outputs is None
            else {
                name: value.detach()
                for name, value in collective_revision_outputs.items()
            }
        )
        self._last_node_keep = node_keep.detach()
        self._last_child_degree_importance = (
            child_degree_importance.detach()
        )

        support_sequence = trend_sequence[:, :, 0:1]
        unknown_sequence = trend_sequence[:, :, 1:2]
        deny_sequence = trend_sequence[:, :, 2:3]
        return (
            output_log_prob,
            unknown_sequence,
            support_sequence,
            deny_sequence,
        )

    def __repr__(self):
        return self.__class__.__name__
