"""Deterministic, feature-only preprocessing of an event's reply graph.

The default partition compares replies to the *same original parent*.  It does
not replace that rule with a shared parent group, even if two parents are twins.
Grouping never sees an event label or a learned graph representation.
"""

from dataclasses import dataclass, field, replace
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

import torch


DEFAULT_RELATION_MAPPING = {
    "generic": 0,
    "unknown": 1,
    "support": 2,
    "deny": 3,
    "query": 4,
    "comment": 5,
}


@dataclass
class GroupGraph:
    """Single-event partition; ``x`` keeps its original autograd connection.

    Group 0 is the source post, whereas ``root_index`` addresses the original
    nodes in ``x``.  In root-only mode unused nodes have ``node_to_group=-1``.
    Edges contain only original forward relations.  The backbone is responsible
    for creating separately typed inverse relations.
    """

    x: torch.Tensor
    node_to_group: torch.Tensor
    group_members: Tuple[Tuple[int, ...], ...]
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    root_index: int
    graph_id: str
    y: Optional[torch.Tensor] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    relation_mapping: Dict[Any, int] = field(default_factory=dict)

    @property
    def num_groups(self):
        return len(self.group_members)

    @property
    def num_nodes(self):
        return int(self.x.size(0))

    def to(self, device):
        return replace(
            self,
            x=self.x.to(device),
            node_to_group=self.node_to_group.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            y=self.y.to(device) if torch.is_tensor(self.y) else self.y,
        )


def _scalar(value, name):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("{} must identify exactly one node/event".format(name))
        return value.item()
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError("{} must identify exactly one node/event".format(name))
        return value[0]
    return value


def _reply_edges(data, edge_direction):
    for name in (
        "reply_edge_index",
        "directed_reply_edge_index",
        "directed_edge_index",
        "direct_edge_index",
    ):
        edges = getattr(data, name, None)
        if edges is not None:
            return edges, name
    edges = getattr(data, "edge_index", None)
    if edges is None:
        raise ValueError("An explicit reply_edge_index is required (use [2, 0] for no replies)")
    declared = edge_direction or getattr(data, "edge_direction", None)
    if declared != "parent_to_child":
        raise ValueError(
            "edge_index direction is ambiguous; provide genuine reply_edge_index "
            "or explicitly declare edge_direction='parent_to_child'"
        )
    return edges, "edge_index"


def _validated_edges(edges, num_nodes):
    edges = torch.as_tensor(edges).detach().cpu()
    if edges.dim() != 2 or edges.size(0) != 2:
        raise ValueError("Reply edges must have shape [2, E]")
    if edges.dtype.is_floating_point or edges.dtype == torch.bool:
        raise ValueError("Reply edge indices must be integers")
    edges = edges.to(torch.long)
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("Reply edge references a node outside x")
    return edges


def _relation_types(data, edge_source, num_edges, relation_mapping, stance_target):
    if stance_target not in {"generic", "parent", "source"}:
        raise ValueError("stance_target must be 'generic', 'parent', or 'source'")
    declared_mapping = relation_mapping
    if declared_mapping is None:
        declared_mapping = getattr(data, "reply_relation_mapping", None)
    mapping = dict(DEFAULT_RELATION_MAPPING if declared_mapping is None else declared_mapping)
    if not mapping or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in mapping.values()):
        raise ValueError("relation_mapping must map relation names/raw values to nonnegative integer IDs")
    if "generic" not in mapping:
        # A raw-label mapping may still use the fixed generic ID, but generic
        # must be named explicitly to make checkpoints self-describing.
        mapping["generic"] = DEFAULT_RELATION_MAPPING["generic"]

    raw = getattr(data, "reply_edge_type", None)
    relation_source = "reply_edge_type" if raw is not None else "generic"
    if raw is None and stance_target == "parent":
        if edge_source == "directed_edge_index":
            raw = getattr(data, "directed_edge_stance", None)
            if raw is not None:
                relation_source = "directed_edge_stance (declared parent target)"
        elif edge_source in {"reply_edge_index", "directed_reply_edge_index", "direct_edge_index"}:
            raw = getattr(data, "reply_edge_stance", None)
            if raw is not None:
                relation_source = "reply_edge_stance (declared parent target)"
    if raw is None:
        return [mapping["generic"]] * num_edges, mapping, relation_source
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().reshape(-1).tolist()
    else:
        raw = list(raw)
    if len(raw) != num_edges:
        raise ValueError("reply relation labels must align with the chosen directed reply edges")
    declared_ids = set(mapping.values())
    raw_stance = "declared parent target" in relation_source
    result = []
    for value in raw:
        if isinstance(value, str):
            if value not in mapping:
                raise ValueError("Undeclared reply relation {!r}; provide a stable relation_mapping".format(value))
            result.append(mapping[value])
        elif isinstance(value, int) and not isinstance(value, bool):
            if declared_mapping is None:
                raise ValueError("Integer reply relation labels require an explicit stable relation_mapping")
            if value in mapping:
                result.append(mapping[value])
            elif not raw_stance and value in declared_ids:
                result.append(value)
            else:
                message = (
                    "Integer parent stance {!r} requires an explicit raw-value mapping"
                    if raw_stance else "Integer reply relation {!r} is not declared in relation_mapping"
                )
                raise ValueError(message.format(value))
        else:
            raise ValueError("Reply relations must be named strings or declared integer IDs")
    return result, mapping, relation_source


def _cycle_nodes(outgoing, incoming):
    """Iterative Kosaraju SCC traversal; no recursion limit for deep threads."""
    seen, order = set(), []
    for start in range(len(outgoing)):
        if start in seen:
            continue
        stack = [(start, False)]
        while stack:
            node, finishing = stack.pop()
            if finishing:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            stack.extend((other, False) for other in outgoing[node] if other not in seen)
    seen, cyclic = set(), set()
    for start in reversed(order):
        if start in seen:
            continue
        component, stack = [], [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in incoming[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        if len(component) > 1 or start in outgoing[start]:
            cyclic.update(component)
    return cyclic


def _context_keys(feature_keys, incoming, incoming_relations, root, cyclic):
    """Ancestral content hashes aid stable group numbering, never grouping.

    No outgoing degree/count enters these keys, so adding identical leaves does
    not change any existing key.  IDs only settle genuinely identical keys.
    """
    keys = {}

    def digest(value):
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()

    for node in range(len(feature_keys)):
        if node == root or node in cyclic or len(incoming[node]) != 1:
            parents = sorted(set(feature_keys[p] for p in incoming[node]))
            keys[node] = digest((feature_keys[node], parents, sorted(incoming_relations[node])))
    for start in range(len(feature_keys)):
        if start in keys:
            continue
        path, node = [], start
        while node not in keys:
            path.append(node)
            node = next(iter(incoming[node]))
        while path:
            child = path.pop()
            parent = next(iter(incoming[child]))
            keys[child] = digest((keys[parent], sorted(incoming_relations[child]), feature_keys[child]))
    return keys


def _partition_bucket(nodes, feature_keys, normalized, threshold, mode, block_size):
    unique = {}
    for node in nodes:
        unique.setdefault(feature_keys[node], []).append(node)
    ordered = sorted(unique)
    if mode == "exact":
        return [unique[key] for key in ordered], [1.0] * len(ordered)

    representatives = [unique[key][0] for key in ordered]
    vectors = normalized[representatives]
    clusters, minima, cluster_heads = [], [], []
    for current in range(len(representatives)):
        compatible = torch.empty(current, dtype=torch.bool)
        similarities = torch.empty(current, dtype=torch.float64)
        for start in range(0, current, block_size):
            end = min(current, start + block_size)
            sims = torch.mv(vectors[start:end], vectors[current]).clamp(-1.0, 1.0)
            similarities[start:end] = sims
            compatible[start:end] = sims >= threshold
        # Ordered unique features, rather than multiplicity or original IDs,
        # control this deterministic complete-link threshold partition.
        destination = None
        candidate_indices = torch.nonzero(
            compatible[cluster_heads], as_tuple=False
        ).reshape(-1).tolist() if cluster_heads else []
        for index in candidate_indices:
            members = clusters[index]
            if bool(compatible[members].all()):
                destination = index
                minima[index] = min(minima[index], float(similarities[members].min()))
                break
        if destination is None:
            clusters.append([current])
            cluster_heads.append(current)
            minima.append(1.0)
        else:
            clusters[destination].append(current)
    groups = [
        [node for unique_index in cluster for node in unique[ordered[unique_index]]]
        for cluster in clusters
    ]
    return groups, minima


def grouping_cache_key(data, *, dataset="", split="", feature_version="", **settings):
    """Content/settings hash for an external bounded cache (no global storage).

    This preprocessing key has no teacher dependency.  Gain-target caches must
    additionally include the frozen teacher checkpoint hash and training split.
    """
    def stable_metadata(value):
        if isinstance(value, dict):
            return [
                [type(key).__name__, repr(key), stable_metadata(item)]
                for key, item in sorted(value.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
            ]
        if isinstance(value, (tuple, list)):
            return [stable_metadata(item) for item in value]
        return value

    h = hashlib.sha256()
    metadata = {"dataset": dataset, "split": split, "feature_version": feature_version, "settings": settings}
    h.update(json.dumps(stable_metadata(metadata), sort_keys=True, default=str).encode("utf-8"))
    for name in (
        "x", "reply_edge_index", "directed_reply_edge_index", "directed_edge_index",
        "direct_edge_index", "edge_index", "reply_edge_type", "directed_edge_stance",
        "reply_edge_stance", "root_index", "rootindex", "graph_id", "event_id",
        "edge_direction", "reply_relation_mapping",
    ):
        value = getattr(data, name, None)
        h.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            h.update(str((value.dtype, tuple(value.shape))).encode("utf-8"))
            byte_values = value.reshape(-1).view(torch.uint8)
            # Bound temporary Python byte lists even for large feature tensors.
            for start in range(0, byte_values.numel(), 65536):
                h.update(bytes(byte_values[start:start + 65536].tolist()))
        else:
            h.update(repr(stable_metadata(value)).encode("utf-8"))
    return h.hexdigest()


def build_group_graph(
    data,
    threshold=0.95,
    mode="context",
    relation_mapping=None,
    stance_target="generic",
    edge_direction=None,
    similarity_block_size=1024,
):
    """Adapt a single graph and form conservative, deterministic reply groups.

    ``context`` uses same-parent, same-relation complete-link grouping;
    ``exact`` merges only equal nonzero features in those same buckets;
    ``feature_only`` removes the context constraints as an ablation;
    ``singleton`` retains every original node; ``root_only`` retains the source.
    Malformed-parent, cyclic, isolated, and zero-feature replies stay singleton.
    Generic relations are used unless explicit local types or parent-target
    stance labels are supplied.  Source-target stance is never repurposed.
    """
    started = time.perf_counter()
    if mode not in {"context", "exact", "feature_only", "singleton", "root_only"}:
        raise ValueError("Unsupported grouping mode: {}".format(mode))
    if not -1.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [-1, 1]")
    if int(similarity_block_size) <= 0:
        raise ValueError("similarity_block_size must be positive")
    x = getattr(data, "x", None)
    if not torch.is_tensor(x) or x.dim() != 2 or x.size(0) < 1:
        raise ValueError("x must be a nonempty [N, F] tensor")
    if getattr(data, "batch", None) is not None:
        batch = data.batch
        if torch.is_tensor(batch) and batch.numel() and int(batch.max()) != int(batch.min()):
            raise ValueError("build_group_graph accepts one event; first split a PyG Batch")
    features = x.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Fixed node features must be finite")
    num_nodes = x.size(0)
    feature_keys = [tuple(row.tolist()) for row in features]
    raw_edges, edge_source = _reply_edges(data, edge_direction)
    edges = _validated_edges(raw_edges, num_nodes)
    types, mapping, relation_source = _relation_types(
        data, edge_source, edges.size(1), relation_mapping, stance_target
    )
    outgoing = [set() for _ in range(num_nodes)]
    incoming = [set() for _ in range(num_nodes)]
    incoming_relations = [set() for _ in range(num_nodes)]
    edge_triples = list(zip(edges[0].tolist(), edges[1].tolist(), types))
    for parent, child, relation in edge_triples:
        outgoing[parent].add(child)
        incoming[child].add(parent)
        incoming_relations[child].add(relation)
    root_value = getattr(data, "root_index", None)
    if root_value is None:
        root_value = getattr(data, "rootindex", None)
    if root_value is None:
        candidates = [node for node, parents in enumerate(incoming) if not parents]
        if len(candidates) != 1:
            raise ValueError(
                "Cannot infer a unique source post ({} zero-indegree nodes); provide root_index".format(len(candidates))
            )
        root, root_source = candidates[0], "unique_zero_indegree"
    else:
        scalar_root = _scalar(root_value, "root_index")
        if isinstance(scalar_root, bool) or int(scalar_root) != scalar_root:
            raise ValueError("root_index must be an integer")
        root, root_source = int(scalar_root), "explicit"
    if root < 0 or root >= num_nodes:
        raise ValueError("root_index is outside x")
    cyclic = _cycle_nodes(outgoing, incoming)
    multi_parent = {node for node in range(num_nodes) if len(incoming[node]) > 1}
    multi_relation = {node for node in range(num_nodes) if len(incoming_relations[node]) > 1}
    isolated = {node for node in range(num_nodes) if not incoming[node] and node != root}
    norms = torch.linalg.vector_norm(features, dim=1)
    zero_nodes = set(torch.nonzero(norms == 0, as_tuple=False).reshape(-1).tolist())
    normalized = features / norms.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(1)
    invalid = cyclic | multi_parent | multi_relation | isolated | zero_nodes
    context_keys = _context_keys(feature_keys, incoming, incoming_relations, root, cyclic)
    groups, minimum_similarities = [], []
    if mode != "root_only":
        buckets = {}
        for node in range(num_nodes):
            if node == root:
                continue
            if mode == "singleton" or node in invalid:
                groups.append([node])
                minimum_similarities.append(1.0)
                continue
            if mode == "feature_only":
                bucket = ("all",)
            else:
                bucket = (next(iter(incoming[node])), next(iter(incoming_relations[node])))
            buckets.setdefault(bucket, []).append(node)
        for nodes in buckets.values():
            formed, minima = _partition_bucket(
                nodes, feature_keys, normalized, float(threshold), mode, int(similarity_block_size)
            )
            groups.extend(formed)
            minimum_similarities.extend(minima)
    # Content controls the ordinary ordering; IDs only settle identical content
    # in distinct original-parent buckets without illegally merging the buckets.
    records = list(zip(groups, minimum_similarities))
    records.sort(key=lambda record: (
        min(context_keys[node] for node in record[0]),
        tuple(sorted(set(feature_keys[node] for node in record[0]))),
        min(record[0]),
    ))
    members = ((root,),) + tuple(tuple(sorted(group)) for group, _ in records)
    node_to_group = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
    for group_index, nodes in enumerate(members):
        node_to_group[list(nodes)] = group_index
    node_map = node_to_group.detach().cpu().tolist()
    internal_count = 0
    discarded_count = 0
    group_edges = set()
    for parent, child, relation in edge_triples:
        source, target = node_map[parent], node_map[child]
        if source < 0 or target < 0:
            discarded_count += 1
            continue
        internal_count += int(source == target)
        group_edges.add((source, target, relation))
    ordered_edges = sorted(group_edges)
    grouped_edge_index = torch.tensor(
        [[edge[0] for edge in ordered_edges], [edge[1] for edge in ordered_edges]],
        dtype=torch.long, device=x.device,
    )
    edge_type = torch.tensor([edge[2] for edge in ordered_edges], dtype=torch.long, device=x.device)
    graph_id = None
    for name in ("graph_id", "event_id", "tweet_id", "id"):
        graph_id = getattr(data, name, None)
        if graph_id is not None:
            graph_id = str(_scalar(graph_id, name))
            break
    if graph_id is None:
        graph_id = "content:" + grouping_cache_key(data)[:24]
    sizes = [len(group) for group in members]
    diagnostics = {
        "original_num_nodes": num_nodes,
        "original_num_edges": edges.size(1),
        "num_groups": len(members),
        "nonroot_num_groups": len(members) - 1,
        "group_sizes": sizes,
        "compression_ratio": 1.0 - len(members) / num_nodes,
        "singleton_fraction": sum(size == 1 for size in sizes) / len(members),
        "group_min_cosine": [1.0] + [minimum for _, minimum in records],
        "grouping_seconds": time.perf_counter() - started,
        "threshold": float(threshold),
        "mode": mode,
        "edge_source": edge_source,
        "edge_direction": "parent_to_child",
        "root_source": root_source,
        "root_original_index": root,
        "root_has_incoming_edges": bool(incoming[root]),
        "root_is_isolated": not incoming[root] and not outgoing[root],
        "relation_source": relation_source,
        "stance_target": stance_target,
        "uses_parent_stance": "declared parent target" in relation_source,
        "multi_parent_nodes": sorted(multi_parent),
        "cyclic_nodes": sorted(cyclic),
        "multi_relation_nodes": sorted(multi_relation),
        "isolated_nonroot_nodes": sorted(isolated),
        "zero_feature_nodes": sorted(zero_nodes),
        "original_internal_edges": internal_count,
        "internal_edge_policy": "preserve_deduplicated_original_relation",
        "root_only_discarded_edges": discarded_count,
        "deduplicated_num_edges": len(ordered_edges),
        "root_outdegree": len(outgoing[root]),
        "similarity_block_size": int(similarity_block_size),
        "approximate_grouping": False,
    }
    split = getattr(data, "split", None)
    if split is not None:
        diagnostics["split"] = str(_scalar(split, "split"))
    return GroupGraph(
        x=x, node_to_group=node_to_group, group_members=members,
        edge_index=grouped_edge_index, edge_type=edge_type,
        root_index=root, graph_id=graph_id, y=getattr(data, "y", None),
        diagnostics=diagnostics, relation_mapping=mapping,
    )
