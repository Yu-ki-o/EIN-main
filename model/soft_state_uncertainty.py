import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeStanceEstimator(nn.Module):
    """
    Predicts whether a child keeps or flips its parent's root-relative state.

    The estimator is trained from existing LLM-derived edge_stance labels, but
    inference only uses node representations and tree structure.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.max_hop = max(1.0, float(getattr(args, 'max_hop', 1)))
        self.uncertainty_max = max(
            1e-6,
            float(getattr(args, 'edge_uncertainty_max', 8.0)),
        )
        input_dim = hidden_dim * 4 + 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden_dim, 2)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_parent, h_child, depth_child):
        depth = depth_child.to(dtype=h_parent.dtype).view(-1, 1) / self.max_hop
        feat = torch.cat(
            [
                h_parent,
                h_child,
                torch.abs(h_parent - h_child),
                h_parent * h_child,
                depth,
            ],
            dim=-1,
        )
        hidden = self.backbone(feat)
        logits = self.logit_head(hidden)
        uncertainty = F.softplus(self.uncertainty_head(hidden))
        uncertainty = uncertainty.clamp(max=self.uncertainty_max)
        return logits, uncertainty


class RootStateEstimator(nn.Module):
    """
    Predicts a node's direct Support/Denial state relative to the root event.

    This direct evidence can reduce uncertainty for deep nodes when it agrees
    with the path-propagated state, so depth alone does not force uncertainty to
    grow.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.max_hop = max(1.0, float(getattr(args, 'max_hop', 1)))
        self.uncertainty_max = max(
            1e-6,
            float(getattr(args, 'root_uncertainty_max', 8.0)),
        )
        input_dim = hidden_dim * 4 + 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden_dim, 2)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_root, h_node, depth_node):
        depth = depth_node.to(dtype=h_root.dtype).view(-1, 1) / self.max_hop
        feat = torch.cat(
            [
                h_root,
                h_node,
                torch.abs(h_root - h_node),
                h_root * h_node,
                depth,
            ],
            dim=-1,
        )
        hidden = self.backbone(feat)
        logits = self.logit_head(hidden)
        uncertainty = F.softplus(self.uncertainty_head(hidden))
        uncertainty = uncertainty.clamp(max=self.uncertainty_max)
        return logits, uncertainty


class SoftStateTargetBuilder(nn.Module):
    """
    Builds uncertainty-aware soft depth targets from edge stance propagation.

    High-uncertainty nodes are kept in the tree because descendants still need
    their structural path; uncertainty only smooths their state and reduces
    their contribution to depth-level supervision.
    """

    def __init__(self, hidden_dim, max_hop, args=None):
        super().__init__()
        self.max_hop = int(max_hop)
        self.undirected = bool(getattr(args, 'undirected', False))
        self.edge_estimator = EdgeStanceEstimator(hidden_dim, args=args)
        self.root_estimator = RootStateEstimator(hidden_dim, args=args)
        self.lambda_edge = max(
            0.0,
            float(getattr(args, 'lambda_edge_state_aux', 0.05)),
        )
        self.lambda_root = max(
            0.0,
            float(getattr(args, 'lambda_root_state_aux', 0.05)),
        )
        self.edge_uncertainty_reg = max(
            0.0,
            float(getattr(args, 'edge_uncertainty_reg', 0.1)),
        )
        self.root_uncertainty_reg = max(
            0.0,
            float(getattr(args, 'root_uncertainty_reg', 0.1)),
        )
        self.path_decay = min(
            max(float(getattr(args, 'soft_state_path_decay', 0.7)), 0.0),
            1.0,
        )
        self.path_uncertainty_weight = max(
            0.0,
            float(getattr(args, 'soft_state_path_uncertainty_weight', 0.5)),
        )
        self.root_uncertainty_weight = max(
            0.0,
            float(getattr(args, 'soft_state_root_uncertainty_weight', 0.3)),
        )
        self.consistency_weight = max(
            0.0,
            float(getattr(args, 'soft_state_consistency_weight', 0.5)),
        )
        self.entropy_weight = max(
            0.0,
            float(getattr(args, 'soft_state_entropy_weight', 0.5)),
        )
        self.teacher_blend = min(
            max(float(getattr(args, 'soft_state_teacher_blend', 0.5)), 0.0),
            1.0,
        )
        self.label_smoothing = min(
            max(float(getattr(args, 'soft_state_label_smoothing', 0.05)), 0.0),
            0.49,
        )
        self.label_uncertainty = max(
            0.0,
            float(getattr(args, 'soft_state_label_uncertainty', 0.1)),
        )
        self.root_teacher_blend = min(
            max(float(getattr(args, 'soft_state_root_teacher_blend', 0.5)), 0.0),
            1.0,
        )
        self.root_label_uncertainty = max(
            0.0,
            float(getattr(args, 'soft_state_root_label_uncertainty', 0.1)),
        )
        self.eps = 1e-8

    def _zero(self, h):
        return h.new_zeros(())

    def _directed_tree_edges(self, edge_index, edge_stance, batch):
        src, dst = edge_index
        valid = (src != dst) & (batch[src] == batch[dst])

        # Directed datasets already store parent -> child edges. The numeric
        # comment id is not guaranteed to be topological, especially in
        # DRWeibo, so src < dst would silently discard valid branches.
        if self.undirected:
            valid = valid & (src < dst)
        edge_pos = valid.nonzero(as_tuple=False).view(-1)
        src = src[edge_pos]
        dst = dst[edge_pos]

        labels = None
        if edge_stance is not None:
            edge_stance = edge_stance.view(-1).long()
            if edge_stance.numel() == edge_index.size(1):
                labels = edge_stance[edge_pos]
        return src, dst, edge_pos, labels

    def _root_indices(self, data, batch):
        if hasattr(data, 'ptr'):
            return data.ptr[:-1].to(device=batch.device)
        starts = torch.ones(batch.size(0), dtype=torch.bool, device=batch.device)
        starts[1:] = batch[1:] != batch[:-1]
        return starts.nonzero(as_tuple=False).view(-1)

    def _node_depths(self, data, num_nodes, src, dst, batch):
        depth = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=batch.device,
        )
        roots = self._root_indices(data, batch)
        depth[roots] = 0

        if src.numel() == 0:
            return depth

        # Vectorized relaxation avoids per-edge Python loops. We run a fixed
        # number of shallow tree relaxations to avoid GPU->CPU syncs from early
        # stopping checks.
        for _ in range(min(self.max_hop + 1, num_nodes)):
            parent_depth = depth[src]
            valid_parent = parent_depth >= 0
            cand = parent_depth + 1
            update = valid_parent & ((depth[dst] < 0) | (cand < depth[dst]))
            depth[dst[update]] = cand[update]
        return depth

    def _edge_loss(self, logits, uncertainty, labels):
        if labels is None:
            return logits.new_zeros(())
        labels = labels.view(-1).long()
        valid = (labels == 0) | (labels == 1)
        safe_labels = labels.clamp(0, 1)
        ce = F.cross_entropy(logits, safe_labels, reduction='none')
        edge_unc = uncertainty.view(-1)
        loss = torch.exp(-edge_unc) * ce + self.edge_uncertainty_reg * edge_unc
        valid_weight = valid.to(dtype=loss.dtype)
        denom = valid_weight.sum().clamp_min(1.0)
        return self.lambda_edge * (loss * valid_weight).sum() / denom

    def _root_loss(self, logits, uncertainty, labels):
        if labels is None:
            return logits.new_zeros(())
        labels = labels.view(-1).long()
        valid = (labels == 0) | (labels == 1)
        safe_labels = labels.clamp(0, 1)
        ce = F.cross_entropy(logits, safe_labels, reduction='none')
        root_unc = uncertainty.view(-1)
        loss = torch.exp(-root_unc) * ce + self.root_uncertainty_reg * root_unc
        valid_weight = valid.to(dtype=loss.dtype)
        denom = valid_weight.sum().clamp_min(1.0)
        return self.lambda_root * (loss * valid_weight).sum() / denom

    def _js_divergence(self, p, q):
        p = p.clamp_min(self.eps)
        q = q.clamp_min(self.eps)
        m = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p / m).log()).sum(dim=-1)
            + (q * (q / m).log()).sum(dim=-1)
        )
        return js / math.log(2.0)

    def forward(self, data, h):
        batch = data.batch
        edge_index = data.edge_index
        edge_stance = getattr(data, 'edge_stance', None)
        batch_size = int(data.user_state.size(0))
        num_nodes = int(h.size(0))

        src, dst, _, labels = self._directed_tree_edges(edge_index, edge_stance, batch)
        if src.numel() == 0:
            target = h.new_zeros(batch_size, self.max_hop, 3)
            uncertainty = h.new_zeros(batch_size, self.max_hop, 1)
            return target, uncertainty, self._zero(h)

        depth = self._node_depths(data, num_nodes, src, dst, batch)
        depth_child = depth[dst].clamp_min(1).to(dtype=h.dtype)
        logits, edge_uncertainty = self.edge_estimator(h[src], h[dst], depth_child)
        edge_loss = (
            self._edge_loss(logits, edge_uncertainty, labels)
            if self.training
            else self._zero(h)
        )
        roots = self._root_indices(data, batch)
        valid_root_node = (depth >= 1) & (depth <= self.max_hop)
        valid_root_idx = valid_root_node.nonzero(as_tuple=False).view(-1)
        root_for_node = roots[batch[valid_root_idx].long()]
        root_logits, root_uncertainty = self.root_estimator(
            h[root_for_node],
            h[valid_root_idx],
            depth[valid_root_idx].clamp_min(1).to(dtype=h.dtype),
        )
        node_state_labels = getattr(data, 'node_state', None)
        root_labels = None
        if node_state_labels is not None and node_state_labels.numel() == num_nodes:
            root_labels = node_state_labels.view(-1).long()[valid_root_idx]
        root_loss = (
            self._root_loss(root_logits, root_uncertainty, root_labels)
            if self.training
            else self._zero(h)
        )
        aux_loss = edge_loss + root_loss

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            edge_unc = edge_uncertainty.detach().view(-1)
            if self.training and labels is not None and self.teacher_blend > 0:
                labels = labels.view(-1).long()
                valid_label = (labels == 0) | (labels == 1)
                teacher_same = torch.where(
                    labels == 0,
                    probs.new_full(labels.size(), 1.0 - self.label_smoothing),
                    probs.new_full(labels.size(), self.label_smoothing),
                )
                teacher = torch.stack((teacher_same, 1.0 - teacher_same), dim=-1)
                probs = torch.where(
                    valid_label.view(-1, 1),
                    self.teacher_blend * teacher + (1.0 - self.teacher_blend) * probs,
                    probs,
                )
                label_uncertainty = edge_unc.new_full(
                    edge_unc.size(),
                    self.label_uncertainty,
                )
                edge_unc = torch.where(
                    valid_label,
                    self.teacher_blend * label_uncertainty
                    + (1.0 - self.teacher_blend) * edge_unc,
                    edge_unc,
                )

            root_probs = F.softmax(root_logits.detach(), dim=-1)
            root_unc_score = 1.0 - torch.exp(-root_uncertainty.detach().view(-1)).clamp(0.0, 1.0)
            if self.training and root_labels is not None and self.root_teacher_blend > 0:
                valid_root_label = (root_labels == 0) | (root_labels == 1)
                teacher_support = torch.where(
                    root_labels == 0,
                    root_probs.new_full(root_labels.size(), 1.0 - self.label_smoothing),
                    root_probs.new_full(root_labels.size(), self.label_smoothing),
                )
                root_teacher = torch.stack(
                    (teacher_support, 1.0 - teacher_support),
                    dim=-1,
                )
                root_probs = torch.where(
                    valid_root_label.view(-1, 1),
                    self.root_teacher_blend * root_teacher
                    + (1.0 - self.root_teacher_blend) * root_probs,
                    root_probs,
                )
                root_label_unc = root_unc_score.new_full(
                    root_unc_score.size(),
                    self.root_label_uncertainty,
                )
                root_unc_score = torch.where(
                    valid_root_label,
                    self.root_teacher_blend * root_label_unc
                    + (1.0 - self.root_teacher_blend) * root_unc_score,
                    root_unc_score,
                )

            root_probs_by_node = h.new_full((num_nodes, 2), 0.5)
            root_unc_by_node = h.new_ones(num_nodes)
            root_probs_by_node[valid_root_idx] = root_probs
            root_unc_by_node[valid_root_idx] = root_unc_score

            node_state = h.new_full((num_nodes, 2), 0.5)
            path_uncertainty = h.new_zeros(num_nodes)
            state_disagreement = h.new_zeros(num_nodes)
            node_state[roots, 0] = 1.0
            node_state[roots, 1] = 0.0

            uniform = h.new_tensor([0.5, 0.5])
            depth_src = depth[src]
            depth_dst = depth[dst]

            for depth_id in range(1, self.max_hop + 1):
                edge_mask = (depth_dst == depth_id) & (depth_src >= 0)
                edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
                parent = src[edge_ids]
                child = dst[edge_ids]
                q_same = probs[edge_ids, 0:1]
                q_diff = probs[edge_ids, 1:2]
                p_parent = node_state[parent]
                p_child = torch.cat(
                    (
                        p_parent[:, 0:1] * q_same + p_parent[:, 1:2] * q_diff,
                        p_parent[:, 0:1] * q_diff + p_parent[:, 1:2] * q_same,
                    ),
                    dim=-1,
                )
                confidence = torch.exp(-edge_unc[edge_ids]).clamp(0.0, 1.0).view(-1, 1)
                p_path = confidence * p_child + (1.0 - confidence) * uniform
                root_p = root_probs_by_node[child]
                edge_risk = 1.0 - confidence.view(-1)
                path_uncertainty[child] = (
                    self.path_decay * path_uncertainty[parent]
                    + (1.0 - self.path_decay) * edge_risk
                ).clamp(0.0, 1.0)
                state_disagreement[child] = self._js_divergence(p_path, root_p)

                path_conf = torch.exp(-path_uncertainty[child]).view(-1, 1)
                root_conf = torch.exp(-root_unc_by_node[child]).view(-1, 1)
                alpha = path_conf / (path_conf + root_conf + self.eps)
                fused = alpha * p_path + (1.0 - alpha) * root_p
                node_state[child] = fused / fused.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            valid_node = (depth >= 1) & (depth <= self.max_hop)
            target = h.new_zeros(batch_size, self.max_hop, 3)
            depth_uncertainty = h.new_zeros(batch_size, self.max_hop, 1)
            state_safe = node_state.clamp_min(self.eps)
            entropy = -(state_safe * state_safe.log()).sum(dim=-1) / math.log(2.0)
            node_uncertainty = (
                self.path_uncertainty_weight * path_uncertainty
                + self.root_uncertainty_weight * root_unc_by_node
                + self.consistency_weight * state_disagreement
                + self.entropy_weight * entropy
            ).clamp_min(0.0)

            valid_batch = batch[valid_node].long()
            valid_depth = depth[valid_node].long() - 1
            flat_index = valid_batch * self.max_hop + valid_depth
            flat_size = batch_size * self.max_hop

            weights = torch.exp(-node_uncertainty[valid_node]).view(-1, 1)
            state_sum = h.new_zeros(flat_size, 2)
            weight_sum = h.new_zeros(flat_size, 1)
            uncertainty_sum = h.new_zeros(flat_size, 1)
            count_sum = h.new_zeros(flat_size, 1)

            state_sum.index_add_(0, flat_index, weights * node_state[valid_node])
            weight_sum.index_add_(0, flat_index, weights)
            uncertainty_sum.index_add_(
                0,
                flat_index,
                node_uncertainty[valid_node].view(-1, 1),
            )
            count_sum.index_add_(
                0,
                flat_index,
                torch.ones_like(node_uncertainty[valid_node]).view(-1, 1),
            )

            state_dist = state_sum / weight_sum.clamp_min(self.eps)
            uncertainty_dist = uncertainty_sum / count_sum.clamp_min(1.0)
            target[:, :, 1:] = state_dist.view(batch_size, self.max_hop, 2)
            depth_uncertainty = uncertainty_dist.view(batch_size, self.max_hop, 1)

            has_nodes = (count_sum.view(batch_size, self.max_hop, 1) > 0)
            target = torch.where(
                has_nodes.expand_as(target),
                target,
                h.new_zeros(target.size()),
            )
            depth_uncertainty = torch.where(
                has_nodes,
                depth_uncertainty,
                h.new_zeros(depth_uncertainty.size()),
            )

        return target, depth_uncertainty, aux_loss
