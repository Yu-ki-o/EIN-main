"""TCSR: Thresholded Collective Stance Revision for rumor detection.

This file implements a runnable PyTorch Geometric prototype. The model keeps
the mechanics explicit so that each intermediate signal can be inspected in
case studies and ablation experiments.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv as PyGGCNConv, global_mean_pool

from model.EIN_ResGCN import GCNConv as ProjectGCNConv


def compute_depth(edge_index, num_nodes, root=0):
    """Compute BFS depth from root on one propagation graph.

    Unreachable nodes are assigned max reachable depth + 1, which keeps them
    visible to depth-wise aggregation without pretending they are close to the
    source post. The frontier expansion stays on edge_index.device, so CUDA
    batches do not bounce through CPU for depth computation.
    """
    device = edge_index.device
    depth = torch.full((int(num_nodes),), -1, dtype=torch.long, device=device)
    if num_nodes <= 0:
        return depth

    if torch.is_tensor(root):
        root = int(root.detach().view(-1)[0].item())
    root = max(0, min(int(root), int(num_nodes) - 1))
    depth[root] = 0

    if edge_index.numel() > 0:
        src, dst = edge_index.long()
        valid_edge = (
            (src >= 0)
            & (src < int(num_nodes))
            & (dst >= 0)
            & (dst < int(num_nodes))
        )
        src = src[valid_edge]
        dst = dst[valid_edge]
        frontier = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
        frontier[root] = True
        for current_depth in range(int(num_nodes)):
            edge_mask = frontier[src]
            if not bool(edge_mask.any()):
                break
            candidate = dst[edge_mask]
            unseen = depth[candidate] < 0
            if not bool(unseen.any()):
                break
            next_nodes = candidate[unseen].unique()
            depth[next_nodes] = current_depth + 1
            frontier = torch.zeros_like(frontier)
            frontier[next_nodes] = True

    reachable = depth >= 0
    if bool(reachable.any()):
        fill_depth = depth[reachable].max() + 1
    else:
        fill_depth = depth.new_tensor(0)
    depth = torch.where(reachable, depth, fill_depth)
    return depth


def _masked_mean(values, mask, dim=1, keepdim=False, eps=1e-6):
    mask = mask.to(dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    total = (values * mask).sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return total / denom


def _masked_max(values, mask, dim=1, keepdim=False):
    mask = mask.to(dtype=torch.bool)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    expanded_mask = mask.expand_as(values)
    fill = torch.finfo(values.dtype).min
    masked = values.masked_fill(~expanded_mask, fill)
    result = masked.max(dim=dim, keepdim=keepdim).values
    has_value = expanded_mask.any(dim=dim, keepdim=keepdim)
    return torch.where(has_value, result, torch.zeros_like(result))


def _depth_weight(num_depths, device, dtype):
    depth = torch.arange(num_depths, device=device, dtype=dtype)
    return 1.0 / (1.0 + depth)


class StanceEstimator(nn.Module):
    """Predict soft support/challenge/uncertain stance probabilities."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, return_logits=False):
        logits = self.network(x.float())
        probs = F.softmax(logits, dim=-1)
        if return_logits:
            return probs, logits
        return probs


def _root_extend(node_features, batch):
    if batch.numel() == 0:
        return node_features
    is_root = torch.ones(
        batch.size(0),
        dtype=torch.bool,
        device=batch.device,
    )
    is_root[1:] = batch[1:] != batch[:-1]
    root_index = is_root.nonzero(as_tuple=False).view(-1)
    return node_features[root_index][batch.long()]


class GraphEncoder(nn.Module):
    """Plain GCN/GAT encoder used as the lightweight TCSR backbone."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        conv_type="gcn",
        gat_heads=2,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        conv_type = str(conv_type).lower()
        in_dim = int(input_dim)
        for _ in range(max(1, int(num_layers))):
            if conv_type == "gat":
                conv = GATConv(
                    in_dim,
                    hidden_dim,
                    heads=max(1, int(gat_heads)),
                    concat=False,
                )
            else:
                conv = PyGGCNConv(in_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = x.float()
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h_graph = global_mean_pool(h, batch)
        return h, h_graph


class BiGCNGraphEncoder(nn.Module):
    """BiGCN-style top-down/bottom-up propagation encoder.

    This mirrors the project's BiGCN backbone idea, but returns node and graph
    representations for the TCSR diagnostic modules instead of classification
    logits.
    """

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.dropout = float(dropout)
        self.td_conv1 = PyGGCNConv(input_dim, hidden_dim)
        self.td_conv2 = PyGGCNConv(hidden_dim + input_dim, hidden_dim)
        self.bu_conv1 = PyGGCNConv(input_dim, hidden_dim)
        self.bu_conv2 = PyGGCNConv(hidden_dim + input_dim, hidden_dim)
        self.td_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bu_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = x.float()
        td_nodes = self._direction_forward(
            x,
            edge_index,
            batch,
            self.td_conv1,
            self.td_conv2,
            self.td_projection,
        )
        reverse_edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0)
        bu_nodes = self._direction_forward(
            x,
            reverse_edge_index,
            batch,
            self.bu_conv1,
            self.bu_conv2,
            self.bu_projection,
        )
        h = self.fusion(torch.cat((td_nodes, bu_nodes), dim=-1))
        h_graph = global_mean_pool(h, batch)
        return h, h_graph

    def _direction_forward(self, x, edge_index, batch, conv1, conv2, projection):
        first_hidden = conv1(x, edge_index)
        hidden = torch.cat((first_hidden, _root_extend(x, batch)), dim=-1)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = torch.cat((hidden, _root_extend(first_hidden, batch)), dim=-1)
        return F.relu(projection(hidden))


class ResGCNGraphEncoder(nn.Module):
    """ResGCN-style residual graph encoder following the project backbone."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        residual=True,
        edge_norm=True,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.bn_feat = nn.BatchNorm1d(input_dim)
        self.conv_feat = ProjectGCNConv(input_dim, hidden_dim, gfn=True)
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(max(1, int(num_layers))):
            self.bns_conv.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(
                ProjectGCNConv(
                    hidden_dim,
                    hidden_dim,
                    edge_norm=edge_norm,
                )
            )
        self.output_norm = nn.LayerNorm(hidden_dim)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = self._safe_batch_norm(self.bn_feat, x.float())
        h = F.relu(self.conv_feat(h, edge_index))
        for batch_norm, conv in zip(self.bns_conv, self.convs):
            update = self._safe_batch_norm(batch_norm, h)
            update = F.relu(conv(update, edge_index))
            update = F.dropout(update, p=self.dropout, training=self.training)
            h = h + update if self.residual else update
        h = self.output_norm(h)
        h_graph = global_mean_pool(h, batch)
        return h, h_graph

    @staticmethod
    def _safe_batch_norm(batch_norm, x):
        if batch_norm.training and x.size(0) <= 1:
            return F.batch_norm(
                x,
                batch_norm.running_mean,
                batch_norm.running_var,
                batch_norm.weight,
                batch_norm.bias,
                training=False,
                eps=batch_norm.eps,
            )
        return batch_norm(x)


def build_graph_encoder(
    input_dim,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2,
    conv_type="gcn",
    gat_heads=2,
    resgcn_residual=True,
    edge_norm=True,
):
    conv_type = str(conv_type).strip().lower()
    if conv_type in {"bigcn", "bi-gcn", "bi_gcn"}:
        return BiGCNGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if conv_type in {"resgcn", "res-gcn", "res_gcn"}:
        return ResGCNGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual=resgcn_residual,
            edge_norm=edge_norm,
        )
    return GraphEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        gat_heads=gat_heads,
    )


class DepthStateAggregator(nn.Module):
    """Aggregate collective stance state at each propagation depth."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, h, stance_probs, depth, batch, batch_size=None):
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
        if depth.numel() == 0:
            num_depths = 1
        else:
            num_depths = int(depth.max().item()) + 1
        num_depths = max(1, num_depths)

        hidden_dim = h.size(-1)
        dtype = h.dtype
        depth = depth.long().clamp_min(0)
        flat_index = batch.long() * num_depths + depth
        flat_size = int(batch_size) * num_depths

        ones = h.new_ones(h.size(0))
        count_flat = h.new_zeros(flat_size)
        count_flat.index_add_(0, flat_index, ones)
        count = count_flat.view(batch_size, num_depths)
        mask = count > 0

        stance_sum = h.new_zeros(flat_size, 3)
        stance_sum.index_add_(0, flat_index, stance_probs.to(dtype=dtype))
        B = stance_sum.view(batch_size, num_depths, 3)
        B = B / count.clamp_min(1.0).unsqueeze(-1)

        class_embs = []
        class_masses = []
        for class_id in range(3):
            weight = stance_probs[:, class_id].to(dtype=dtype).unsqueeze(-1)
            emb_sum = h.new_zeros(flat_size, hidden_dim)
            emb_sum.index_add_(0, flat_index, h * weight)
            mass = h.new_zeros(flat_size)
            mass.index_add_(0, flat_index, weight.squeeze(-1))
            emb = emb_sum / mass.clamp_min(self.eps).unsqueeze(-1)
            class_embs.append(emb.view(batch_size, num_depths, hidden_dim))
            class_masses.append(mass.view(batch_size, num_depths))

        return {
            "B": B,
            "count": count,
            "support_emb": class_embs[0],
            "challenge_emb": class_embs[1],
            "uncertain_emb": class_embs[2],
            "support_mass": class_masses[0],
            "challenge_mass": class_masses[1],
            "uncertain_mass": class_masses[2],
            "mask": mask,
        }


class ReinforcementEstimator(nn.Module):
    """Estimate social reinforcement from support state and coherence."""

    def __init__(self, hidden_dim=64, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.scorer = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, B, support_emb, root_emb, mask):
        support = B[..., 0]
        challenge = B[..., 1]
        coherence = F.cosine_similarity(
            support_emb,
            root_emb.unsqueeze(1),
            dim=-1,
            eps=self.eps,
        )
        coherence = (coherence + 1.0) * 0.5
        dominance = (support / (challenge + self.eps)).clamp(max=10.0) / 10.0
        depth_weight = _depth_weight(B.size(1), B.device, B.dtype)
        depth_weight = depth_weight.view(1, -1).expand_as(support)
        features = torch.stack(
            (support, coherence, dominance, depth_weight),
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        return scores * mask.to(dtype=scores.dtype)


class CorrectionPressureEstimator(nn.Module):
    """Estimate correction pressure from stance, structure, and semantics."""

    def __init__(self, hidden_dim=64, use_semantics=True, eps=1e-6):
        super().__init__()
        self.use_semantics = bool(use_semantics)
        self.eps = float(eps)
        self.scorer = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        B,
        count,
        support_emb,
        challenge_emb,
        uncertain_emb,
        root_emb,
        mask,
        return_components=False,
    ):
        support = B[..., 0]
        challenge = B[..., 1]
        uncertain = B[..., 2]
        total_nodes = count.sum(dim=1, keepdim=True).clamp_min(1.0)
        volume = torch.log1p(count) / torch.log1p(total_nodes)
        depth_weight = _depth_weight(B.size(1), B.device, B.dtype)
        depth_weight = depth_weight.view(1, -1).expand_as(challenge)
        centrality_proxy = depth_weight

        root_context = root_emb.unsqueeze(1).expand_as(challenge_emb)
        root_contrast = 0.5 * (
            1.0
            - F.cosine_similarity(
                challenge_emb,
                root_context,
                dim=-1,
                eps=self.eps,
            )
        )
        support_contrast = 0.5 * (
            1.0
            - F.cosine_similarity(
                challenge_emb,
                support_emb,
                dim=-1,
                eps=self.eps,
            )
        )
        semantic_strength = challenge * torch.maximum(
            root_contrast,
            support_contrast,
        )
        challenge_dominance = (
            challenge / (support + uncertain + self.eps)
        ).clamp(max=10.0) / 10.0

        if not self.use_semantics:
            root_contrast = torch.zeros_like(root_contrast)
            support_contrast = torch.zeros_like(support_contrast)
            semantic_strength = torch.zeros_like(semantic_strength)
            challenge_dominance = torch.zeros_like(challenge_dominance)

        features = torch.stack(
            (
                challenge,
                volume,
                depth_weight,
                challenge * centrality_proxy,
                semantic_strength,
                support_contrast * challenge,
                uncertain,
                challenge_dominance,
            ),
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        scores = scores * mask.to(dtype=scores.dtype)
        if not return_components:
            return scores
        return scores, {
            "root_contrast": root_contrast * mask.to(dtype=root_contrast.dtype),
            "support_contrast": support_contrast
            * mask.to(dtype=support_contrast.dtype),
            "semantic_strength": semantic_strength
            * mask.to(dtype=semantic_strength.dtype),
            "challenge_dominance": challenge_dominance
            * mask.to(dtype=challenge_dominance.dtype),
        }


class RevisionOpportunityMask(nn.Module):
    """Mark depths with enough future nodes to observe revision response."""

    def __init__(self, window_k=2, min_future_nodes=1):
        super().__init__()
        self.window_k = max(1, int(window_k))
        self.min_future_nodes = max(1, int(min_future_nodes))

    def forward(self, count):
        future_count = count.new_zeros(count.size())
        for offset in range(1, self.window_k + 1):
            if offset >= count.size(1):
                break
            future_count[:, :-offset] += count[:, offset:]
        has_current_depth = count > 0
        return ((future_count >= self.min_future_nodes) & has_current_depth).to(
            dtype=count.dtype
        )


class CorrectionResistantAnomaly(nn.Module):
    """Observed correction failure: challenge pressure followed by support."""

    def __init__(self, window_k=2, pool="mean", eps=1e-6):
        super().__init__()
        self.window_k = max(1, int(window_k))
        self.pool = str(pool)
        self.eps = float(eps)

    def forward(self, B, count, pressure, opportunity_mask, expected_future_B=None):
        support = B[..., 0]
        future_B_sum = B.new_zeros(B.size())
        future_count = count.new_zeros(count.size())
        for offset in range(1, self.window_k + 1):
            if offset >= support.size(1):
                break
            future_B_sum[:, :-offset] += B[:, offset:] * count[
                :, offset:
            ].unsqueeze(-1)
            future_count[:, :-offset] += count[:, offset:]
        future_B = future_B_sum / future_count.clamp_min(self.eps).unsqueeze(-1)

        if expected_future_B is None:
            expected_support = support
            revision_residual = future_B - B
        else:
            expected_support = expected_future_B[..., 0]
            revision_residual = future_B - expected_future_B

        support_future = future_B[..., 0]
        anomaly = opportunity_mask * pressure * F.relu(
            support_future - expected_support
        )
        revision_residual = revision_residual * opportunity_mask.unsqueeze(-1)

        valid = opportunity_mask > 0
        if self.pool == "max":
            pooled = _masked_max(anomaly, valid, dim=1, keepdim=True)
        else:
            pooled = _masked_mean(anomaly, valid, dim=1, keepdim=True, eps=self.eps)
        residual_graph = _masked_mean(
            revision_residual,
            valid,
            dim=1,
            eps=self.eps,
        )
        return anomaly, pooled, future_B, revision_residual, residual_graph


class RevisionExpectationPredictor(nn.Module):
    """Counterfactual stance expectation under reinforcement/correction."""

    def __init__(self, graph_dim, hidden_dim=64, dropout=0.1, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gate_mlp = nn.Sequential(
            nn.Linear(graph_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, B, reinforcement, pressure, h_graph, mask):
        batch_size, num_depths, _ = B.size()
        graph_context = h_graph.unsqueeze(1).expand(
            batch_size,
            num_depths,
            h_graph.size(-1),
        )
        depth_weight = _depth_weight(num_depths, B.device, B.dtype)
        depth_weight = depth_weight.view(1, num_depths, 1).expand(
            batch_size,
            num_depths,
            1,
        )
        features = torch.cat(
            (
                B,
                reinforcement.unsqueeze(-1),
                pressure.unsqueeze(-1),
                depth_weight,
                graph_context,
            ),
            dim=-1,
        )
        gates = self.gate_mlp(features)
        adopt_gain = gates[..., 0]
        revise_gain = gates[..., 1]
        uncertain_gain = gates[..., 2]

        conflict = torch.minimum(reinforcement, pressure)
        delta_support = adopt_gain * reinforcement - revise_gain * pressure
        delta_challenge = revise_gain * pressure - adopt_gain * reinforcement
        delta_uncertain = uncertain_gain * conflict
        deltas = torch.stack(
            (delta_support, delta_challenge, delta_uncertain),
            dim=-1,
        )
        expected = F.softmax(torch.log(B.clamp_min(self.eps)) + deltas, dim=-1)
        expected = expected * mask.to(dtype=expected.dtype).unsqueeze(-1)
        return expected, {
            "adopt_gain": adopt_gain * mask.to(dtype=adopt_gain.dtype),
            "revise_gain": revise_gain * mask.to(dtype=revise_gain.dtype),
            "uncertain_gain": uncertain_gain
            * mask.to(dtype=uncertain_gain.dtype),
        }


class CorrectionIsolation(nn.Module):
    """Challenge isolation: pressure appearing far from the propagation core."""

    def __init__(self, pool="max", eps=1e-6):
        super().__init__()
        self.pool = str(pool)
        self.eps = float(eps)

    def forward(self, pressure, opportunity_mask, depth_mask):
        weights = _depth_weight(pressure.size(1), pressure.device, pressure.dtype)
        weights = weights.view(1, -1).expand_as(pressure)
        isolation = pressure * (1.0 - weights) * (1.0 - opportunity_mask)
        isolation = isolation * depth_mask.to(dtype=pressure.dtype)
        if self.pool == "mean":
            pooled = _masked_mean(
                isolation,
                depth_mask,
                dim=1,
                keepdim=True,
                eps=self.eps,
            )
        else:
            pooled = _masked_max(isolation, depth_mask, dim=1, keepdim=True)
        return isolation, pooled


class ReinforcementDominance(nn.Module):
    """Core dominance of support reinforcement over challenge pressure."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, B, depth_mask):
        weights = _depth_weight(B.size(1), B.device, B.dtype)
        weights = weights.view(1, -1)
        mask = depth_mask.to(dtype=B.dtype)
        core_support = (weights * B[..., 0] * mask).sum(dim=1)
        core_challenge = (weights * B[..., 1] * mask).sum(dim=1)
        dominance = core_support / (core_challenge + self.eps)
        return dominance.unsqueeze(-1)


class AdaptiveThresholdModule(nn.Module):
    """Learn graph-level adoption and revision thresholds."""

    def __init__(self, graph_dim, hidden_dim=64, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.threshold_mlp = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, h_graph, reinforcement, pressure, mask):
        thresholds = self.threshold_mlp(h_graph)
        theta_adopt = thresholds[:, 0:1]
        theta_revise = thresholds[:, 1:2]
        adopt_score = torch.sigmoid(reinforcement - theta_adopt)
        revise_score = torch.sigmoid(pressure - theta_revise)
        valid = mask.to(dtype=reinforcement.dtype)
        adopt_graph = _masked_mean(
            adopt_score,
            valid,
            dim=1,
            keepdim=True,
            eps=self.eps,
        )
        revise_graph = _masked_mean(
            revise_score,
            valid,
            dim=1,
            keepdim=True,
            eps=self.eps,
        )
        asymmetry = adopt_graph - revise_graph
        return {
            "theta_adopt": theta_adopt,
            "theta_revise": theta_revise,
            "adopt_score": adopt_score * valid,
            "revise_score": revise_score * valid,
            "adopt_score_graph": adopt_graph,
            "revise_score_graph": revise_graph,
            "threshold_asymmetry": asymmetry,
        }


class TCSRClassifier(nn.Module):
    """Final rumor classifier over graph and TCSR diagnostic features."""

    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.network(features)


class TCSRModel(nn.Module):
    """Full TCSR model with ablation switches for each diagnostic family."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=2,
        gnn_layers=2,
        dropout=0.2,
        conv_type="gcn",
        gat_heads=2,
        resgcn_residual=True,
        edge_norm=True,
        window_k=2,
        min_future_nodes=1,
        use_anomaly=True,
        use_isolation=True,
        use_dominance=True,
        use_threshold=True,
        use_revision_expectation=True,
        use_correction_semantics=True,
        use_external_stance=True,
        eps=1e-6,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.use_anomaly = bool(use_anomaly)
        self.use_isolation = bool(use_isolation)
        self.use_dominance = bool(use_dominance)
        self.use_threshold = bool(use_threshold)
        self.use_revision_expectation = bool(use_revision_expectation)
        self.use_correction_semantics = bool(use_correction_semantics)
        self.use_external_stance = bool(use_external_stance)
        self.eps = float(eps)

        self.stance_estimator = StanceEstimator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.graph_encoder = build_graph_encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            conv_type=conv_type,
            gat_heads=gat_heads,
            resgcn_residual=resgcn_residual,
            edge_norm=edge_norm,
        )
        self.depth_aggregator = DepthStateAggregator(eps=eps)
        self.reinforcement_estimator = ReinforcementEstimator(eps=eps)
        self.pressure_estimator = CorrectionPressureEstimator(
            use_semantics=use_correction_semantics,
            eps=eps,
        )
        self.opportunity_masker = RevisionOpportunityMask(
            window_k=window_k,
            min_future_nodes=min_future_nodes,
        )
        self.revision_predictor = RevisionExpectationPredictor(
            graph_dim=hidden_dim,
            eps=eps,
        )
        self.anomaly_module = CorrectionResistantAnomaly(
            window_k=window_k,
            eps=eps,
        )
        self.isolation_module = CorrectionIsolation(eps=eps)
        self.dominance_module = ReinforcementDominance(eps=eps)
        self.threshold_module = AdaptiveThresholdModule(
            graph_dim=hidden_dim,
            eps=eps,
        )

        classifier_dim = hidden_dim + 6 + 11
        self.classifier = TCSRClassifier(
            input_dim=classifier_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index.long()
        batch = self._batch_vector(data, x)
        batch_size = self._batch_size(data, batch)

        h, h_graph = self.graph_encoder(x, edge_index, batch)
        stance_probs, stance_logits = self._resolve_stance(data, x)
        depth = self._resolve_depth(data, edge_index, batch, batch_size)
        roots = self._root_indices(data, batch, batch_size)
        root_emb = h[roots]

        depth_state = self.depth_aggregator(
            h=h,
            stance_probs=stance_probs,
            depth=depth,
            batch=batch,
            batch_size=batch_size,
        )
        B = depth_state["B"]
        count = depth_state["count"]
        depth_mask = depth_state["mask"]

        reinforcement = self.reinforcement_estimator(
            B,
            depth_state["support_emb"],
            root_emb,
            depth_mask,
        )
        pressure, pressure_components = self.pressure_estimator(
            B,
            count,
            depth_state["support_emb"],
            depth_state["challenge_emb"],
            depth_state["uncertain_emb"],
            root_emb,
            depth_mask,
            return_components=True,
        )
        opportunity_mask = self.opportunity_masker(count)

        expected_future_B = None
        revision_gates = self._empty_revision_gates(
            pressure,
            batch_size,
            pressure.size(1),
        )
        if self.use_revision_expectation:
            expected_future_B, revision_gates = self.revision_predictor(
                B,
                reinforcement,
                pressure,
                h_graph,
                depth_mask,
            )

        (
            raw_anomaly,
            raw_anomaly_graph,
            future_B,
            raw_revision_residual,
            raw_revision_residual_graph,
        ) = self.anomaly_module(
            B,
            count,
            pressure,
            opportunity_mask,
            expected_future_B=expected_future_B,
        )
        if self.use_anomaly:
            anomaly = raw_anomaly
            anomaly_graph = raw_anomaly_graph
            revision_residual = raw_revision_residual
            revision_residual_graph = raw_revision_residual_graph
        else:
            anomaly = pressure.new_zeros(pressure.size())
            anomaly_graph = pressure.new_zeros(batch_size, 1)
            revision_residual = pressure.new_zeros(B.size())
            revision_residual_graph = pressure.new_zeros(batch_size, 3)

        if self.use_isolation:
            isolation, isolation_graph = self.isolation_module(
                pressure,
                opportunity_mask,
                depth_mask,
            )
        else:
            isolation = pressure.new_zeros(pressure.size())
            isolation_graph = pressure.new_zeros(batch_size, 1)

        if self.use_dominance:
            dominance_graph = self.dominance_module(B, depth_mask)
        else:
            dominance_graph = pressure.new_zeros(batch_size, 1)

        threshold_outputs = self._threshold_outputs(
            h_graph,
            reinforcement,
            pressure,
            depth_mask,
        )

        B_mean = _masked_mean(B, depth_mask, dim=1, eps=self.eps)
        B_max = _masked_max(B, depth_mask, dim=1)
        reinforcement_graph = _masked_mean(
            reinforcement,
            depth_mask,
            dim=1,
            keepdim=True,
            eps=self.eps,
        )
        pressure_graph = _masked_mean(
            pressure,
            depth_mask,
            dim=1,
            keepdim=True,
            eps=self.eps,
        )

        features = torch.cat(
            (
                h_graph,
                B_mean,
                B_max,
                reinforcement_graph,
                pressure_graph,
                anomaly_graph,
                revision_residual_graph,
                isolation_graph,
                dominance_graph,
                threshold_outputs["theta_adopt"],
                threshold_outputs["theta_revise"],
                threshold_outputs["threshold_asymmetry"],
            ),
            dim=-1,
        )
        logits = self.classifier(features)

        aux_outputs = {
            "stance_probs": stance_probs,
            "stance_logits": stance_logits,
            "depth": depth,
            "B": B,
            "count": count,
            "mask": depth_mask,
            "support_emb": depth_state["support_emb"],
            "challenge_emb": depth_state["challenge_emb"],
            "uncertain_emb": depth_state["uncertain_emb"],
            "R": reinforcement,
            "P": pressure,
            "R_graph": reinforcement_graph,
            "P_graph": pressure_graph,
            "correction_pressure_components": pressure_components,
            "expected_future_B": expected_future_B,
            "future_B": future_B,
            "revision_residual": revision_residual,
            "revision_residual_graph": revision_residual_graph,
            "revision_gates": revision_gates,
            "A_obs": anomaly,
            "A_obs_graph": anomaly_graph,
            "A_iso": isolation,
            "A_iso_graph": isolation_graph,
            "A_dom_graph": dominance_graph,
            "theta_adopt": threshold_outputs["theta_adopt"],
            "theta_revise": threshold_outputs["theta_revise"],
            "adopt_score": threshold_outputs["adopt_score"],
            "revise_score": threshold_outputs["revise_score"],
            "adopt_score_graph": threshold_outputs["adopt_score_graph"],
            "revise_score_graph": threshold_outputs["revise_score_graph"],
            "threshold_asymmetry": threshold_outputs["threshold_asymmetry"],
            "opportunity_mask": opportunity_mask,
            "graph_embedding": h_graph,
        }
        return logits, aux_outputs

    def compute_loss(
        self,
        logits,
        data,
        aux_outputs,
        stance_loss_weight=1.0,
        revision_loss_weight=0.0,
    ):
        return compute_tcsr_loss(
            logits,
            data,
            aux_outputs,
            stance_loss_weight=stance_loss_weight,
            revision_loss_weight=revision_loss_weight,
        )

    @staticmethod
    def _empty_revision_gates(reference, batch_size, num_depths):
        zeros = reference.new_zeros(batch_size, num_depths)
        return {
            "adopt_gain": zeros,
            "revise_gain": zeros,
            "uncertain_gain": zeros,
        }

    def _threshold_outputs(self, h_graph, reinforcement, pressure, depth_mask):
        if self.use_threshold:
            return self.threshold_module(
                h_graph,
                reinforcement,
                pressure,
                depth_mask,
            )

        valid = depth_mask.to(dtype=reinforcement.dtype)
        batch_size = h_graph.size(0)
        theta_adopt = h_graph.new_full((batch_size, 1), 0.5)
        theta_revise = h_graph.new_full((batch_size, 1), 0.5)
        adopt_score = reinforcement * valid
        revise_score = pressure * valid
        adopt_graph = _masked_mean(
            adopt_score,
            depth_mask,
            dim=1,
            keepdim=True,
            eps=self.eps,
        )
        revise_graph = _masked_mean(
            revise_score,
            depth_mask,
            dim=1,
            keepdim=True,
            eps=self.eps,
        )
        return {
            "theta_adopt": theta_adopt,
            "theta_revise": theta_revise,
            "adopt_score": adopt_score,
            "revise_score": revise_score,
            "adopt_score_graph": adopt_graph,
            "revise_score_graph": revise_graph,
            "threshold_asymmetry": h_graph.new_zeros(batch_size, 1),
        }

    def _resolve_stance(self, data, x):
        stance_probs = getattr(data, "stance_probs", None)
        if self.use_external_stance and stance_probs is not None:
            stance_probs = stance_probs.to(device=x.device, dtype=x.dtype)
            stance_probs = stance_probs.view(-1, 3)
            stance_probs = stance_probs.clamp_min(self.eps)
            stance_probs = stance_probs / stance_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
            return stance_probs, None
        return self.stance_estimator(x, return_logits=True)

    def _resolve_depth(self, data, edge_index, batch, batch_size):
        depth = getattr(data, "depth", None)
        if depth is not None:
            depth = depth.to(device=edge_index.device).view(-1).long()
            if depth.numel() == batch.numel():
                return self._fill_negative_depth(depth, batch, batch_size)
        return self._compute_batch_depth(edge_index, batch, data, batch_size)

    def _fill_negative_depth(self, depth, batch, batch_size):
        depth = depth.clone()
        for graph_id in range(batch_size):
            node_mask = batch == graph_id
            graph_depth = depth[node_mask]
            if graph_depth.numel() == 0:
                continue
            reachable = graph_depth >= 0
            fill = (
                graph_depth[reachable].max() + 1
                if bool(reachable.any())
                else graph_depth.new_tensor(0)
            )
            depth[node_mask & (depth < 0)] = fill
        return depth.clamp_min(0)

    def _compute_batch_depth(self, edge_index, batch, data, batch_size):
        roots = self._root_indices(data, batch, batch_size)
        depth = torch.zeros(batch.size(0), dtype=torch.long, device=batch.device)
        ptr = getattr(data, "ptr", None)
        if ptr is not None:
            ptr = ptr.to(device=batch.device)
            for graph_id in range(batch_size):
                start = int(ptr[graph_id].item())
                end = int(ptr[graph_id + 1].item())
                edge_mask = (
                    (edge_index[0] >= start)
                    & (edge_index[0] < end)
                    & (edge_index[1] >= start)
                    & (edge_index[1] < end)
                )
                local_edge = edge_index[:, edge_mask] - start
                local_root = int(roots[graph_id].item()) - start
                depth[start:end] = compute_depth(
                    local_edge,
                    end - start,
                    root=local_root,
                )
            return depth

        local_id = torch.full_like(batch, -1)
        for graph_id in range(batch_size):
            node_ids = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            local_id[node_ids] = torch.arange(
                node_ids.numel(),
                device=batch.device,
                dtype=torch.long,
            )
            edge_mask = (batch[edge_index[0]] == graph_id) & (
                batch[edge_index[1]] == graph_id
            )
            local_edge = local_id[edge_index[:, edge_mask]]
            local_root = int(local_id[roots[graph_id]].item())
            depth[node_ids] = compute_depth(
                local_edge,
                node_ids.numel(),
                root=local_root,
            )
        return depth

    def _root_indices(self, data, batch, batch_size):
        root_index = getattr(data, "root_index", None)
        if root_index is not None:
            root_index = root_index.to(device=batch.device).view(-1).long()
            if root_index.numel() == batch_size:
                return root_index
            if root_index.numel() == 1 and batch_size == 1:
                return root_index

        ptr = getattr(data, "ptr", None)
        if ptr is not None:
            return ptr[:-1].to(device=batch.device).long()

        roots = []
        for graph_id in range(batch_size):
            node_ids = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            roots.append(node_ids[0])
        return torch.stack(roots).long()

    @staticmethod
    def _batch_vector(data, x):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        return batch.long()

    @staticmethod
    def _batch_size(data, batch):
        num_graphs = getattr(data, "num_graphs", None)
        if num_graphs is not None:
            return int(num_graphs)
        return int(batch.max().item()) + 1 if batch.numel() else 1


def compute_tcsr_loss(
    logits,
    data,
    aux_outputs=None,
    stance_loss_weight=1.0,
    revision_loss_weight=0.0,
    ignore_index=-100,
):
    """Cross entropy rumor loss with optional stance/revision supervision."""
    labels = data.y.view(-1).long().to(device=logits.device)
    classification_loss = F.cross_entropy(logits, labels)
    loss = classification_loss
    components = {
        "classification_loss": classification_loss.detach(),
        "stance_loss": logits.new_zeros(()).detach(),
        "revision_loss": logits.new_zeros(()).detach(),
    }

    stance_labels = getattr(data, "stance_labels", None)
    if stance_labels is not None and aux_outputs is not None:
        stance_labels = stance_labels.view(-1).long().to(device=logits.device)
        valid = stance_labels != int(ignore_index)
        valid = valid & (stance_labels >= 0) & (stance_labels < 3)
        if bool(valid.any()):
            stance_logits = aux_outputs.get("stance_logits")
            if stance_logits is not None:
                stance_loss = F.cross_entropy(
                    stance_logits[valid],
                    stance_labels[valid],
                )
            else:
                stance_probs = aux_outputs["stance_probs"].clamp_min(1e-8)
                stance_loss = F.nll_loss(
                    stance_probs[valid].log(),
                    stance_labels[valid],
                )
            loss = loss + float(stance_loss_weight) * stance_loss
            components["stance_loss"] = stance_loss.detach()

    if (
        aux_outputs is not None
        and float(revision_loss_weight) > 0.0
        and aux_outputs.get("expected_future_B") is not None
    ):
        expected_future = aux_outputs["expected_future_B"]
        future = aux_outputs["future_B"]
        opportunity = aux_outputs["opportunity_mask"] > 0
        if bool(opportunity.any()):
            revision_loss = F.mse_loss(
                expected_future[opportunity],
                future[opportunity].detach(),
            )
            loss = loss + float(revision_loss_weight) * revision_loss
            components["revision_loss"] = revision_loss.detach()

    components["total_loss"] = loss.detach()
    return loss, components
