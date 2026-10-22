"""Stance-response trajectories with sufficiency-constrained path selection.

Paths follow original parent-to-child edges; stance is local to the parent.
Shared prefixes are encoded once. Binary Concrete gates relax path selection
during training; evaluation physically removes unselected nodes BEFORE attention.
The selector's set summary is never passed to the event classifier.
"""

from dataclasses import dataclass, fields
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ReplyTree:
    x: torch.Tensor
    y: torch.Tensor | None
    root: int
    parent: torch.Tensor
    relation: torch.Tensor
    depth: torch.Tensor
    branch: torch.Tensor
    entry: torch.Tensor
    exit: torch.Tensor
    levels: tuple
    paths: tuple
    path_nodes: torch.Tensor
    path_index: torch.Tensor

    def to(self, device):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device)
            elif isinstance(value, tuple):
                value = tuple(item.to(device) for item in value)
            values[field.name] = value
        return ReplyTree(**values)


def _checked_stance_labels(labels, count, field):
    if labels is None:
        raise ValueError(
            "{} is missing: each original reply edge needs a local stance label. "
            "Use a cache containing directed_edge_index/directed_edge_stance or "
            "edge_index/edge_stance built from annotated stance_label values.".format(field))
    if not torch.is_tensor(labels) or labels.numel() != count:
        actual = labels.numel() if torch.is_tensor(labels) else "not a tensor"
        raise ValueError("{} has {} labels for {} matching edges; rebuild the inconsistent "
                         "cache from annotated raw data.".format(field, actual, count))
    labels = labels.detach().cpu().reshape(-1)
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("{} requires complete local stance labels: 0=support, 1=deny; "
                         "annotate missing stance_label values before rebuilding the cache.".format(field))
    return labels.long()


def _aligned_reply_stances(data, edges, num_nodes):
    """Keep original reply edges and recover legacy labels by ordered endpoints.

    Older EIN caches store directed_edge_index but only edge_stance aligned to
    edge_index. Those edge lists may differ in both order and size. Never zip
    the two unrelated tensors, or reverse a parent-relative stance relation.
    An explicitly present directed_edge_stance remains authoritative.
    """
    directed = getattr(data, "directed_edge_index", None) is not None
    field = "directed_edge_stance" if directed else "edge_stance"
    labels = getattr(data, field, None)
    if labels is not None or not directed:
        return _checked_stance_labels(labels, edges.size(1), field)

    legacy_edges = getattr(data, "edge_index", None)
    legacy_labels = getattr(data, "edge_stance", None)
    if legacy_edges is None or legacy_labels is None:
        raise ValueError(
            "directed_edge_stance is missing and legacy edge_index/edge_stance is incomplete "
            "(edge_index={}, edge_stance={}). Rebuild the cache from raw data with "
            "parent-relative stance_label annotations.".format(
                "missing" if legacy_edges is None else "present",
                "missing" if legacy_labels is None else "present"))
    if (not torch.is_tensor(legacy_edges) or legacy_edges.ndim != 2
            or legacy_edges.size(0) != 2 or legacy_edges.dtype != torch.long):
        raise ValueError("Legacy edge_index for edge_stance alignment must be a long [2,E] tensor")
    legacy_edges = legacy_edges.detach().cpu()
    if legacy_edges.numel() and (legacy_edges.min() < 0 or legacy_edges.max() >= num_nodes):
        raise ValueError("Legacy edge_index for edge_stance alignment contains an invalid node index")
    legacy_labels = _checked_stance_labels(legacy_labels, legacy_edges.size(1), "edge_stance")
    by_edge = {}
    for endpoints, label in zip(legacy_edges.t().tolist(), legacy_labels.tolist()):
        edge = tuple(endpoints)
        if edge in by_edge and by_edge[edge] != label:
            raise ValueError("Conflicting legacy edge_stance labels for reply edge {}".format(edge))
        by_edge[edge] = label
    reply_edges = [tuple(pair) for pair in edges.t().tolist()]
    missing = [edge for edge in reply_edges if edge not in by_edge]
    if missing:
        raise ValueError(
            "Legacy edge_index/edge_stance lacks labels for {} original directed reply edges "
            "(examples: {}); rebuild an aligned cache. Reverse-edge labels are not substituted.".format(
                len(missing), missing[:3]))
    return torch.tensor([by_edge[edge] for edge in reply_edges], dtype=torch.long)


def prepare_reply_tree(data, use_stance=True):
    """Validate a rooted reply tree, without inventing edges or stance labels.

    EIN caches put the source at node zero and store directed_edge_index plus
    aligned directed_edge_stance. Legacy edge_index/edge_stance can supply
    labels by matching endpoints while retaining the original directed tree.
    Plain edge_index alone is accepted only if it is already a directed rooted
    tree. Invalid/missing parent relations fail fast.
    """
    if isinstance(data, ReplyTree):
        return data
    x = data.x.detach().cpu()
    if x.ndim != 2 or x.size(0) == 0 or not torch.isfinite(x).all():
        raise ValueError("StancePath requires finite features for at least one node")
    n = x.size(0)
    root_value = getattr(data, "root_index", None)
    if root_value is None:
        root_value = getattr(data, "rootindex", 0)
    root_tensor = torch.as_tensor(root_value).reshape(-1)
    if root_tensor.numel() != 1 or root_tensor.item() != int(root_tensor.item()):
        raise ValueError("root_index must be one integer local node index")
    root = int(root_tensor.item())
    if not 0 <= root < n:
        raise ValueError("root_index is outside the graph")
    edges = getattr(data, "directed_edge_index", None)
    if edges is None:
        edges = getattr(data, "edge_index", None)
    if edges is None and n == 1:
        edges = torch.empty((2, 0), dtype=torch.long)
    if edges is None or edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("Original parent-to-child directed_edge_index is required")
    edges = edges.detach().cpu()
    if edges.dtype != torch.long:
        raise ValueError("Reply edge indices must have dtype torch.long")
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Reply edge contains an invalid node index")
    if use_stance and edges.size(1):
        stance = _aligned_reply_stances(data, edges, n)
    else:
        stance = torch.zeros(edges.size(1), dtype=torch.long)
    parent, relation = [-1] * n, [2] * n  # 2 is the START marker, root only.
    children = [[] for _ in range(n)]
    seen_edges = {}
    for (p, child), label in zip(edges.t().tolist(), stance.tolist()):
        label = int(label)
        if (p, child) in seen_edges:
            if seen_edges[p, child] != label:
                raise ValueError("Duplicate reply edges have conflicting stance labels")
            continue
        seen_edges[p, child] = label
        if p == child or child == root or parent[child] != -1:
            raise ValueError("Expected a rooted tree with exactly one parent per reply")
        parent[child], relation[child] = p, label
        children[p].append(child)
    depth, branch = [0] * n, [root] * n
    levels, order = [[root]], [root]
    while levels[-1]:
        next_level = []
        for p in levels[-1]:
            for child in sorted(children[p]):
                depth[child] = depth[p] + 1
                branch[child] = child if p == root else branch[p]
                next_level.append(child)
        if not next_level:
            break
        order.extend(next_level)
        levels.append(next_level)
    if len(order) != n:
        raise ValueError("All replies must be reachable from the source; cycle/disconnected tree")
    entry, exit_time, clock = [0] * n, [0] * n, 0
    stack = [(root, False)]
    while stack:
        node, closing = stack.pop()
        if closing:
            exit_time[node] = clock
            continue
        entry[node], clock = clock, clock + 1
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(sorted(children[node])))
    paths = []
    for leaf in order:
        if leaf == root or children[leaf]:
            continue
        path, node = [], leaf
        while node != root:
            path.append(node)
            node = parent[node]
        paths.append(torch.tensor(path[::-1], dtype=torch.long))
    path_nodes = torch.cat(paths) if paths else torch.empty(0, dtype=torch.long)
    path_index = torch.repeat_interleave(torch.arange(len(paths)),
                                         torch.tensor([len(p) for p in paths], dtype=torch.long))
    y = getattr(data, "y", None)
    return ReplyTree(
        x=x, y=y.detach().cpu() if y is not None else None, root=root,
        parent=torch.tensor(parent), relation=torch.tensor(relation),
        depth=torch.tensor(depth), branch=torch.tensor(branch),
        entry=torch.tensor(entry), exit=torch.tensor(exit_time),
        levels=tuple(torch.tensor(level) for level in levels), paths=tuple(paths),
        path_nodes=path_nodes, path_index=path_index,
    )


def depth_encoding(depth, width, dtype):
    frequencies = torch.exp(torch.arange(0, width, 2, device=depth.device).float()
                            * (-math.log(10000.0) / width))
    angles = depth.float().unsqueeze(1) * frequencies
    result = angles.new_zeros((depth.numel(), width))
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :width // 2].cos()
    return result.to(dtype)


class TrajectoryAttention(nn.Module):
    def __init__(self, width, heads, dropout, structural_bias, depth_clip):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.structural_bias, self.depth_clip = structural_bias, depth_clip
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.gate = nn.Linear(2 * width, width)
        self.dropout = nn.Dropout(dropout)
        if structural_bias:
            self.depth_bias = nn.Embedding(2 * depth_clip + 1, heads)
            self.ancestor_bias = nn.Embedding(4, heads)
            self.branch_bias = nn.Embedding(2, heads)

    def forward(self, tokens, weights, tree, node_ids):
        count, width = tokens.shape
        q, k, v = self.qkv(tokens).reshape(count, 3, self.heads, self.head_dim).unbind(1)
        scores = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(self.head_dim)
        if self.structural_bias:
            depth = tree.depth[node_ids]
            diff = (depth[:, None] - depth[None, :]).clamp(-self.depth_clip, self.depth_clip)
            entry, end = tree.entry[node_ids], tree.exit[node_ids]
            ancestor = (entry[:, None] <= entry[None, :]) & (end[:, None] > entry[None, :])
            relation = ancestor.long() + 2 * ancestor.t().long()
            branch = tree.branch[node_ids]
            same_branch = (branch[:, None] == branch[None, :]).long()
            bias = (self.depth_bias(diff + self.depth_clip)
                    + self.ancestor_bias(relation) + self.branch_bias(same_branch))
            scores = scores + bias.permute(2, 0, 1)
        # Gate keys before interaction, never only the final readout. Zero-weight
        # nodes have already been removed by the caller, including at inference.
        scores = scores + weights.clamp_min(1e-8).log()[None, None, :]
        context = torch.einsum("hij,jhd->ihd", self.dropout(scores.softmax(-1)), v)
        context = self.output(context.reshape(count, width))
        return tokens + torch.sigmoid(self.gate(torch.cat((tokens, context), -1))) * context


class StancePath(nn.Module):
    """Return event logits; use return_details for masks and compression metrics."""
    def __init__(self, in_feats, hidden_dim=128, num_classes=2, args=None):
        super().__init__()
        get = lambda name, default: getattr(args, "stance_path_" + name, default)
        self.in_feats, self.hidden_dim, self.num_classes = in_feats, hidden_dim, num_classes
        self.use_stance = bool(get("use_stance", True))
        self.use_selection = bool(get("use_selection", True))
        self.temperature = float(get("temperature", 0.67))
        self.threshold = float(get("threshold", 0.5))
        self.prior = float(get("prior", 0.2))
        self.sufficiency_weight = float(get("sufficiency_weight", 1.0))
        self.sufficiency_margin = float(get("sufficiency_margin", 0.05))
        self.kl_weight = float(get("kl_weight", 0.001))
        self.kl_reduction = str(get("kl_reduction", "sum"))
        heads, layers = int(get("heads", 4)), int(get("interaction_layers", 1))
        depth_clip = int(get("depth_clip", 16))
        dropout = float(getattr(args, "dropout", 0.2))
        if min(in_feats, hidden_dim, num_classes, heads, depth_clip) < 1 or hidden_dim % heads:
            raise ValueError("Positive dimensions required; hidden_dim must be divisible by heads")
        if layers < 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid attention layer count or dropout")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("stance_path_temperature must be positive and finite")
        if not 0 < self.prior < 1 or not 0 <= self.threshold <= 1:
            raise ValueError("prior must be in (0,1), threshold in [0,1]")
        if any(not math.isfinite(v) or v < 0 for v in (
                self.sufficiency_weight, self.sufficiency_margin, self.kl_weight)):
            raise ValueError("Loss weights and sufficiency margin must be finite and nonnegative")
        if self.kl_reduction not in {"sum", "mean"}:
            raise ValueError("stance_path_kl_reduction must be sum or mean")
        width = hidden_dim
        self.text_projection = nn.Linear(in_feats, width)
        self.source_state = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.stance_embedding = nn.Embedding(2, width)
        self.transition_embedding = nn.Embedding(6, width)  # (previous including START, current)
        self.interaction = nn.Sequential(nn.Linear(4 * width, width), nn.GELU())
        self.field = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(width, width), nn.Tanh())
        self.update_gate = nn.Linear(2 * width, width)
        self.token_projection = nn.Linear(2 * width, width)
        self.path_score = nn.Linear(width, 1)
        self.selector = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        # Start with q=0.5; the sparse prior acts through the KL rather than
        # prematurely suppressing the teacher-initialized trajectory encoder.
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        self.attention = nn.ModuleList([
            TrajectoryAttention(width, heads, dropout, bool(get("structural_bias", True)), depth_clip)
            for _ in range(layers)
        ])
        self.event_score = nn.Linear(width, 1)
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(width, num_classes))

    def prepare(self, data):
        if isinstance(data, (list, tuple)):
            graphs = data
        elif callable(getattr(data, "to_data_list", None)):
            graphs = data.to_data_list()
        else:
            graphs = [data]
        return [prepare_reply_tree(graph, self.use_stance) for graph in graphs]

    def encode_trajectory(self, tree):
        # Fixed input encoder features; trainable projection/state updates.
        x = self.text_projection(tree.x.detach())
        h = x.new_zeros(x.shape)
        root_id = tree.levels[0]
        h = h.index_copy(0, root_id, self.source_state(x[root_id]))
        delta = x.new_zeros(x.shape)
        for nodes in tree.levels[1:]:
            parents = tree.parent[nodes]
            relations = tree.relation[nodes]
            stance = self.stance_embedding(relations) if self.use_stance else torch.zeros_like(x[nodes])
            transition = (self.transition_embedding(tree.relation[parents] * 2 + relations)
                          if self.use_stance else torch.zeros_like(stance))
            interaction = self.interaction(torch.cat((
                x[root_id].expand(nodes.numel(), -1), x[parents], x[nodes], stance), -1))
            previous = h[parents]
            change = self.field(torch.cat((previous, interaction, transition), -1))
            change = torch.sigmoid(self.update_gate(torch.cat((previous, interaction), -1))) * change
            h = h.index_copy(0, nodes, previous + change)
            delta = delta.index_copy(0, nodes, change)
        tokens = self.token_projection(torch.cat((h, delta), -1))
        return tokens + depth_encoding(tree.depth, self.hidden_dim, tokens.dtype)

    def path_probabilities(self, tokens, tree):
        if not tree.paths:
            return tokens.new_empty(0)
        values = tokens[tree.path_nodes]
        scores = self.path_score(values).squeeze(-1)
        maximum = scores.new_full((len(tree.paths),), -torch.inf)
        maximum = maximum.scatter_reduce(0, tree.path_index, scores, reduce="amax", include_self=True)
        weights = (scores - maximum[tree.path_index]).exp()
        denominator = weights.new_zeros(len(tree.paths)).scatter_add(0, tree.path_index, weights)
        weights = weights / denominator[tree.path_index]
        summary = tokens.new_zeros((len(tree.paths), self.hidden_dim)).index_add(
            0, tree.path_index, weights[:, None] * values)
        overview = summary.mean(0, keepdim=True).expand_as(summary)
        return self.selector(torch.cat((summary, overview), -1)).squeeze(-1).sigmoid()

    def sample_paths(self, probabilities):
        if not self.training:
            return (probabilities >= self.threshold).to(probabilities.dtype)
        uniform = torch.rand_like(probabilities).clamp(1e-6, 1 - 1e-6)
        noise = uniform.log() - torch.log1p(-uniform)
        return ((torch.logit(probabilities.clamp(1e-6, 1 - 1e-6)) + noise)
                / self.temperature).sigmoid()

    def forward(self, data, *, full_paths=False, source_only=False,
                path_masks=None, return_details=False):
        trees = self.prepare(data)
        if not trees:
            raise ValueError("An empty graph batch is not supported")
        if path_masks is not None and len(path_masks) != len(trees):
            raise ValueError("Supply one path mask per event")
        device = next(self.parameters()).device
        outputs, details = [], []
        for index, raw_tree in enumerate(trees):
            tree = raw_tree.to(device)
            tokens = self.encode_trajectory(tree)
            q = self.path_probabilities(tokens, tree)
            if source_only:
                mask = torch.zeros_like(q)
            elif path_masks is not None:
                mask = torch.as_tensor(path_masks[index], device=device, dtype=tokens.dtype)
                if mask.shape != q.shape or not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
                    raise ValueError("Each path mask must match the path count with values in [0,1]")
            elif full_paths or not self.use_selection:
                mask = torch.ones_like(q)
            else:
                mask = self.sample_paths(q)
            # Noisy-OR is exact set union for binary masks and a differentiable
            # relaxation for Concrete masks. Shared nodes occur exactly once.
            excluded = tokens.new_ones(tokens.size(0)).scatter_reduce(
                0, tree.path_nodes, 1 - mask[tree.path_index], reduce="prod", include_self=True)
            node_weights = 1 - excluded
            node_weights = node_weights.index_fill(0, tree.levels[0], 1.0)
            node_ids = (node_weights.detach() > 0).nonzero(as_tuple=False).flatten()
            selected, weights = tokens[node_ids], node_weights[node_ids]
            for attention in self.attention:
                selected = attention(selected, weights, tree, node_ids)
            pool_weights = (self.event_score(selected).squeeze(-1) + weights.clamp_min(1e-8).log()).softmax(0)
            outputs.append(self.classifier((pool_weights[:, None] * selected).sum(0)))
            safe_q = q.clamp(1e-6, 1 - 1e-6)
            kl = (safe_q * (safe_q.log() - math.log(self.prior))
                  + (1 - safe_q) * (torch.log1p(-safe_q) - math.log1p(-self.prior)))
            kl = kl.mean() if self.kl_reduction == "mean" and q.numel() else kl.sum()
            details.append(dict(probabilities=q, path_mask=mask, node_weights=node_weights,
                                selected_node_ids=node_ids, mask_kl=kl,
                                expected_paths=q.sum(), num_paths=q.numel(), num_nodes=tokens.size(0)))
        logits = torch.stack(outputs)
        return (logits, details) if return_details else logits

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr,
                                weight_decay=getattr(args, "weight_decay", 0.0))


def compute_stance_path_loss(logits, target, details, teacher_logits=None, *,
                            sufficiency_weight=1.0, sufficiency_margin=0.05, kl_weight=0.001):
    """Mean over events of CE + lambda*[CE - stopgrad(teacher CE) - eps]+ + beta*KL."""
    target = target.reshape(-1).long()
    ce = F.cross_entropy(logits, target, reduction="none")
    if teacher_logits is None:
        if sufficiency_weight > 0:
            raise ValueError("A frozen full-path teacher is required for the sufficiency loss")
        sufficiency = ce.new_zeros(ce.shape)
    else:
        teacher_ce = F.cross_entropy(teacher_logits.detach(), target, reduction="none")
        sufficiency = (ce - teacher_ce - sufficiency_margin).relu()
    kl = torch.stack([item["mask_kl"] for item in details])
    loss = (ce + sufficiency_weight * sufficiency + kl_weight * kl).mean()
    return loss, dict(classification=ce.mean(), sufficiency=sufficiency.mean(), mask_kl=kl.mean())
