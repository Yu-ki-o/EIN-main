"""DIGNN-inspired, inductive graph classifier for rumor detection.

The default topology branch uses fixed directed stance-WL patterns, graph-local
self-attention and differentiable hierarchical soft combinations. Version one
trains with classification alone, without contrastive or information losses.
The older energy/structural encoders remain explicit legacy modes.
See docs/DIGNN_WL.md for the representation, configuration and limitations.
"""

from __future__ import annotations

import math
import hashlib
from collections import Counter, OrderedDict, deque

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import batched_negative_sampling, coalesce, to_undirected


def normalized_hsic(topology, text):
    """Normalized, biased RBF HSIC on paired event embeddings.

    Uses detached median squared-distance bandwidths and centered kernels.
    Returns zero for constant views or fewer than three events: for B=2,
    normalized centered kernels are always aligned and provide no useful
    independence signal. This is a dependence penalty, not an MI estimate.
    """
    if topology.ndim != 2 or topology.shape != text.shape:
        raise ValueError("HSIC views must have matching [B, H] shapes")
    zero = (topology.sum() + text.sum()) * 0.0
    if topology.size(0) < 3:
        return zero

    def centered_kernel(z):
        distances = torch.cdist(z, z).square()
        upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        nonzero = distances.detach()[upper & (distances.detach() > 1e-8)]
        if nonzero.numel() == 0:
            return distances * 0.0
        bandwidth = nonzero.median().clamp_min(1e-8)
        kernel = torch.exp(-distances / (2 * bandwidth))
        return (kernel - kernel.mean(0, keepdim=True)
                - kernel.mean(1, keepdim=True) + kernel.mean())

    ka, kx = centered_kernel(topology), centered_kernel(text)
    denominator = (ka.square().sum() * kx.square().sum()).clamp_min(1e-12).sqrt()
    return ((ka * kx).sum() / denominator).clamp(0.0, 1.0)


class StanceEnergyField(nn.Module):
    """Unroll a fixed, strongly convex two-channel relation energy.

    E(phi) = fidelity/2 * ||phi-source||^2
           + tau/2 * sum_e ||A_parent[r] phi_parent
                                - A_child[r] phi_child||^2.

    Positive bounded diagonal maps are learned for support/deny/unknown and
    parent/child roles. Denial never prescribes a sign flip or channel swap.
    Each event has its own safe step size, so batching cannot change a field.
    """

    def __init__(self, scales=(0.25, 1.0, 4.0), num_steps=8,
                 fidelity=1.0, map_bound=0.5, step_fraction=0.9):
        super().__init__()
        if isinstance(scales, str):
            scales = scales.split(",")
        try:
            scales = tuple(float(value) for value in scales)
        except (TypeError, ValueError) as error:
            raise ValueError("dignn_energy_scales must be a sequence of positive numbers") from error
        if not scales or any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("dignn_energy_scales must be finite and positive")
        if isinstance(num_steps, bool) or int(num_steps) != num_steps or num_steps < 1:
            raise ValueError("dignn_energy_steps must be a positive integer")
        if not math.isfinite(fidelity) or fidelity <= 0:
            raise ValueError("dignn_energy_fidelity must be finite and positive")
        if not math.isfinite(map_bound) or not 0 <= map_bound <= 2:
            raise ValueError("dignn_energy_map_bound must be finite and in [0, 2]")
        if not math.isfinite(step_fraction) or not 0 < step_fraction <= 1:
            raise ValueError("dignn_energy_step_fraction must be in (0, 1]")
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.num_steps = int(num_steps)
        self.fidelity = float(fidelity)
        self.map_bound = float(map_bound)
        self.step_fraction = float(step_fraction)
        self.relation_log_scales = nn.Parameter(torch.zeros(3, 2, 2))

    def relation_scales(self):
        return torch.exp(self.map_bound * self.relation_log_scales.tanh())

    def _energy(self, fields, source, edges, relation, parent_map, child_map, batch):
        n, count = source.size(0), int(batch[-1]) + 1
        scales = self.scales.to(fields).view(1, -1)
        source_energy = 0.5 * self.fidelity * (fields - source[:, None]).square().sum(-1)
        residual = (fields[edges[0]] * parent_map[:, None]
                    - fields[edges[1]] * child_map[:, None])
        edge_energy = 0.5 * scales * residual.square().sum(-1)
        # Half of each edge's energy belongs to each endpoint. Keep relation
        # type and endpoint role separate, including explicitly unknown edges.
        local = fields.new_zeros(n * 6, self.scales.numel())
        local = local.index_add(0, edges[0] * 6 + relation * 2, 0.5 * edge_energy)
        local = local.index_add(0, edges[1] * 6 + relation * 2 + 1, 0.5 * edge_energy)
        local = local.view(n, 3, 2, -1).permute(0, 3, 1, 2)
        totals = fields.new_zeros(count, self.scales.numel()).index_add(
            0, batch, source_energy + local.sum(dim=(-1, -2)))
        return source_energy, local, totals

    def forward(self, source, edges, stance, batch):
        if (source.ndim != 2 or source.size(0) == 0 or source.size(1) != 2
                or not source.is_floating_point()
                or not bool(torch.isfinite(source).all()) or bool((source < 0).any())):
            raise ValueError("energy source must be a finite nonnegative [N, 2] tensor")
        n = source.size(0)
        if (batch.dtype != torch.long or batch.shape != (n,)
                or batch.device != source.device or int(batch[0]) != 0
                or bool((batch[1:] < batch[:-1]).any())):
            raise ValueError("energy batch must be sorted long [N] indices starting at zero")
        count = int(batch[-1]) + 1
        if bool((torch.bincount(batch, minlength=count) == 0).any()):
            raise ValueError("energy batch cannot contain empty events")
        if (edges.ndim != 2 or edges.size(0) != 2 or edges.dtype != torch.long
                or edges.device != source.device):
            raise ValueError("energy edges must be a long [2, E] tensor on the source device")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
            raise ValueError("energy edge endpoint is outside source")
        if bool((batch[edges[0]] != batch[edges[1]]).any()):
            raise ValueError("energy edges cannot connect different events")
        if (stance.dtype != torch.long or stance.shape != (edges.size(1),)
                or stance.device != source.device
                or not bool(((stance == -1) | (stance == 0) | (stance == 1)).all())):
            raise ValueError("energy stance must be aligned long [E] labels in {-1, 0, 1}")
        relation = torch.where(stance < 0, 2, stance)
        maps = self.relation_scales().to(source)
        parent_map, child_map = maps[relation, 0], maps[relation, 1]
        scales = self.scales.to(source).view(1, -1, 1)
        fields = source[:, None].expand(-1, self.scales.numel(), -1)
        _, _, initial_energy = self._energy(
            fields, source, edges, relation, parent_map, child_map, batch)

        # Rayleigh bound: ||A_p phi_p-A_c phi_c||^2 <=
        # 2||A_p||^2||phi_p||^2 + 2||A_c||^2||phi_c||^2.
        incident = source.new_zeros(n).index_add(0, edges[0], parent_map.square().amax(-1))
        incident = incident.index_add(0, edges[1], child_map.square().amax(-1))
        event_bound = source.new_zeros(count).scatter_reduce(
            0, batch, incident, reduce="amax", include_self=True)
        bound = self.fidelity + 2 * scales * event_bound[batch, None, None]
        step = self.step_fraction / bound
        for _ in range(self.num_steps):
            residual = (fields[edges[0]] * parent_map[:, None]
                        - fields[edges[1]] * child_map[:, None])
            gradient = self.fidelity * (fields - source[:, None])
            gradient = gradient.index_add(0, edges[0], scales * parent_map[:, None] * residual)
            gradient = gradient.index_add(0, edges[1], -scales * child_map[:, None] * residual)
            fields = F.relu(fields - step * gradient)
        source_energy, relation_energy, final_energy = self._energy(
            fields, source, edges, relation, parent_map, child_map, batch)
        return dict(fields=fields, residuals=fields - source[:, None],
                    source_energy=source_energy, relation_energy=relation_energy,
                    initial_energy=initial_energy, final_energy=final_energy)


class StructureEncoder(nn.Module):
    """Rooted GIN on structure/energy descriptors, with no text input."""

    def __init__(self, hidden_dim, num_layers, dropout, input_dim=6):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim)), train_eps=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, descriptors, edges):
        hidden = F.gelu(self.projection(descriptors))
        for conv, norm in zip(self.convs, self.norms):
            update = F.gelu(conv(hidden, edges))
            hidden = norm(hidden + F.dropout(update, self.dropout, self.training))
        return hidden


class StanceWLTokenizer:
    """Fixed, inductive WL features; never build a vocabulary from a batch.

    Full 128-bit digests identify types before embedding lookup. Two halves
    address separate embedding tables, so a single bucket collision does not
    merge types. This is feature hashing, not an exact learned WL dictionary.
    Incoming and outgoing labeled multisets are hashed separately. Only roots,
    edges and parent-relative stance enter the signatures; node IDs do not.
    """

    def __init__(self, rounds=2, buckets=8192, cache_size=256):
        if rounds < 1 or buckets < 2 or cache_size < 0:
            raise ValueError("WL requires rounds >= 1, buckets >= 2 and cache_size >= 0")
        self.rounds, self.buckets, self.cache_size = rounds, buckets, cache_size
        self._cache = OrderedDict()

    @staticmethod
    def _digest(value):
        return hashlib.blake2b(repr(value).encode("ascii"), digest_size=16).digest()

    def _patterns(self, n, root, pairs):
        key = (n, root, tuple(sorted(pairs)))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        incoming, outgoing = [[] for _ in range(n)], [[] for _ in range(n)]
        for parent, child, relation in key[2]:
            incoming[child].append((relation, parent))
            outgoing[parent].append((relation, child))
        labels = [self._digest(("root" if v == root else "node",)) for v in range(n)]
        patterns = []
        for radius in range(self.rounds + 1):
            for signature, count in sorted(Counter(labels).items()):
                patterns.append((int.from_bytes(signature[:8], "big") % self.buckets,
                                 int.from_bytes(signature[8:], "big") % self.buckets,
                                 radius, count))
            if radius < self.rounds:
                labels = [self._digest((
                    labels[v],
                    tuple(sorted((r, labels[u]) for r, u in incoming[v])),
                    tuple(sorted((r, labels[w]) for r, w in outgoing[v]))))
                    for v in range(n)]
        if self.cache_size:
            self._cache[key] = patterns
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return patterns

    def __call__(self, edges, stance, batch, roots):
        # One host transfer per tensor, rather than synchronization per node.
        event_ids = batch.detach().cpu().tolist()
        sizes = torch.bincount(batch).cpu().tolist()
        root_ids = roots.detach().cpu().tolist()
        edge_list = edges.detach().cpu().t().tolist()
        relations = stance.detach().cpu().tolist()
        offsets, offset = [], 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        grouped = [[] for _ in sizes]
        for (parent, child), relation in zip(edge_list, relations):
            event = event_ids[parent]
            start = offsets[event]
            grouped[event].append((parent - start, child - start, relation))
        patterns = [self._patterns(size, root - start, pairs)
                    for size, root, start, pairs in zip(sizes, root_ids, offsets, grouped)]
        width = max(map(len, patterns))
        packed = torch.zeros(len(sizes), width, 4, dtype=torch.long)
        for i, rows in enumerate(patterns):
            packed[i, :len(rows)] = torch.tensor(rows, dtype=torch.long)
        packed = packed.to(batch.device)
        return dict(ids=packed[..., :2], scales=packed[..., 2],
                    counts=packed[..., 3], valid=packed[..., 3] > 0)


class SubstructureCombination(nn.Module):
    """Within-graph attention followed by mass-preserving soft clustering."""

    def __init__(self, hidden_dim, heads, clusters, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))
        self.assignment = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, clusters))
        self.mass_projection = nn.Linear(2, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.clusters = clusters

    def forward(self, hidden, mass, valid, log_attention=False):
        normalized = self.norm1(hidden)
        update, attention = self.attention(
            normalized, normalized, normalized, key_padding_mask=~valid,
            need_weights=log_attention, average_attn_weights=True)
        hidden = hidden + self.dropout(update)
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        hidden = hidden.masked_fill(~valid[..., None], 0.0)
        # Each graph may use fewer slots; padding and other events cannot
        # increase its number of composite structures.
        capacity = valid.sum(-1).clamp_max(self.clusters)
        next_valid = torch.arange(self.clusters, device=hidden.device)[None] < capacity[:, None]
        logits = self.assignment(hidden).masked_fill(~next_valid[:, None], float("-inf"))
        assignment = logits.softmax(-1) * valid[..., None]
        weighted = assignment * mass[..., None]
        next_mass = weighted.sum(1)
        combined = weighted.transpose(1, 2) @ hidden
        combined = combined / next_mass.clamp_min(1e-8)[..., None]
        statistics = torch.stack((next_mass.log1p(),
                                  next_mass / next_mass.sum(-1, keepdim=True).clamp_min(1)), -1)
        combined = self.output_norm(combined + self.mass_projection(statistics))
        combined = combined.masked_fill(~next_valid[..., None], 0.0)
        if attention is not None:
            attention = attention.masked_fill(~valid[..., None], 0.0)
        return combined, next_mass, next_valid, assignment, attention


class WLTopologyEncoder(nn.Module):
    """Inductive substructure encoder with no text or cross-graph attention."""

    def __init__(self, hidden_dim, layers, dropout, args):
        super().__init__()
        rounds = int(getattr(args, "dignn_wl_rounds", 2))
        buckets = int(getattr(args, "dignn_wl_buckets", 8192))
        heads = int(getattr(args, "dignn_wl_heads", 4))
        if heads < 1 or hidden_dim % heads:
            raise ValueError("dignn_wl_heads must be positive and divide hidden_dim")
        clusters = getattr(args, "dignn_wl_clusters", None)
        if clusters is None:
            clusters = [max(1, 64 // (2 ** i)) for i in range(layers)]
        if isinstance(clusters, str):
            clusters = clusters.split(",")
        clusters = tuple(int(value) for value in clusters)
        if not clusters or any(value < 1 for value in clusters):
            raise ValueError("dignn_wl_clusters must contain positive slot counts")
        if any(b > a for a, b in zip(clusters, clusters[1:])):
            raise ValueError("dignn_wl_clusters must be nonincreasing")
        self.tokenizer = StanceWLTokenizer(rounds, buckets, int(getattr(args, "dignn_wl_cache_size", 256)))
        self.type_embeddings = nn.ModuleList([nn.Embedding(buckets, hidden_dim) for _ in range(2)])
        self.scale_embedding = nn.Embedding(rounds + 1, hidden_dim)
        self.count_projection = nn.Linear(2, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            SubstructureCombination(hidden_dim, heads, count, dropout) for count in clusters])
        self.readout_score = nn.Linear(hidden_dim, 1)
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.log_attention = bool(getattr(args, "dignn_wl_log_attention", False))

    def forward(self, edges, stance, batch, roots):
        tokens = self.tokenizer(edges, stance, batch, roots)
        mass = tokens["counts"].to(self.count_projection.weight.dtype)
        valid = tokens["valid"]
        statistics = torch.stack((mass.log1p(), mass / mass.sum(-1, keepdim=True)), -1)
        hidden = (self.type_embeddings[0](tokens["ids"][..., 0])
                  + self.type_embeddings[1](tokens["ids"][..., 1])
                  + self.scale_embedding(tokens["scales"]) + self.count_projection(statistics))
        hidden = self.input_norm(hidden).masked_fill(~valid[..., None], 0.0)
        diagnostics = {"wl_token_count": valid.sum(-1).detach()}
        for level, block in enumerate(self.blocks):
            hidden, mass, valid, assignment, attention = block(hidden, mass, valid, self.log_attention)
            diagnostics[f"wl_level_{level}_token_count"] = valid.sum(-1).detach()
            if self.log_attention:
                diagnostics[f"wl_level_{level}_assignment"] = assignment.detach()
                diagnostics[f"wl_level_{level}_attention"] = attention.detach()
        scores = self.readout_score(hidden).squeeze(-1) + mass.clamp_min(1e-8).log()
        weights = scores.masked_fill(~valid, float("-inf")).softmax(-1)
        graph = self.readout((hidden * weights[..., None]).sum(1))
        if self.log_attention:
            diagnostics["wl_readout_attention"] = weights.detach()
        return dict(topology_graph=graph, topology_tokens=hidden,
                    topology_token_mass=mass, topology_token_mask=valid,
                    wl_tokens=tokens, wl_diagnostics=diagnostics)


class DIGNN(nn.Module):
    """Return (log_probs, U, S, D) for EINTrainer.

    Inputs: x [N, in_feats], edge_index [2, E], optional directed_edge_index,
    PyG batch, and optional globally indexed root_index/rootindex [B]. In the
    project's cache the source is the first node of each event. WL/energy mode
    reads aligned parent-relative edge stances (0=support, 1=deny, -1=unknown).
    Rumor labels, node/user states and timestamps never construct node inputs.
    """

    def __init__(self, in_feats, hidden_dim=128, num_classes=2, args=None):
        super().__init__()
        self.args = args
        self.in_feats = int(in_feats)
        self.hidden_dim = int(hidden_dim)
        self.max_hop = int(getattr(args, "max_hop", 1))
        layers = int(getattr(args, "dignn_num_layers", getattr(args, "n_layers_conv", 3)))
        dropout = float(getattr(args, "dropout", 0.3))
        self.fusion = str(getattr(args, "dignn_fusion", "attention")).strip().lower()
        self.topology_features = str(getattr(args, "dignn_topology_features", "wl")).strip().lower()
        self.loss_weights = {
            "text_reconstruction": float(getattr(args, "dignn_text_recon_weight", 0.1)),
            "structure_reconstruction": float(getattr(args, "dignn_structure_recon_weight", 0.1)),
            "edge_reconstruction": float(getattr(args, "dignn_edge_recon_weight", 0.1)),
            "independence": float(getattr(args, "dignn_independence_weight", 0.01)),
        }
        # Version one is classification-only, even when old YAML files contain
        # reconstruction/HSIC weights. Legacy modes still honor those options.
        if self.topology_features == "wl":
            self.loss_weights = {name: 0.0 for name in self.loss_weights}
        if min(self.in_feats, self.hidden_dim, num_classes, self.max_hop, layers) < 1:
            raise ValueError("dimensions, classes, max_hop and layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.fusion not in {"attention", "mean", "topology", "text"}:
            raise ValueError("dignn_fusion must be attention, mean, topology or text")
        if self.topology_features not in {"wl", "energy", "structural"}:
            raise ValueError("dignn_topology_features must be wl, energy or structural")
        if any(not math.isfinite(w) or w < 0 for w in self.loss_weights.values()):
            raise ValueError("DIGNN loss weights must be finite and nonnegative")

        self.energy_field = None
        self.structure_input_dim = 6
        if self.topology_features == "energy":
            self.energy_field = StanceEnergyField(
                scales=getattr(args, "dignn_energy_scales", (0.25, 1.0, 4.0)),
                num_steps=getattr(args, "dignn_energy_steps", 8),
                fidelity=float(getattr(args, "dignn_energy_fidelity", 1.0)),
                map_bound=float(getattr(args, "dignn_energy_map_bound", 0.5)),
                step_fraction=float(getattr(args, "dignn_energy_step_fraction", 0.9)))
            self.structure_input_dim = 16
        topology_dim = (self.structure_input_dim if self.energy_field is None else
                        self.structure_input_dim + 11 * self.energy_field.scales.numel())
        self.topology_encoder = (WLTopologyEncoder(self.hidden_dim, layers, dropout, args)
                                 if self.topology_features == "wl" else
                                 StructureEncoder(self.hidden_dim, layers, dropout, topology_dim))
        self.text_encoder = nn.Sequential(
            nn.Linear(self.in_feats, self.hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        # Root semantics and whole-event content are retained within each view.
        self.topology_readout = None if self.topology_features == "wl" else self._readout()
        self.text_readout = self._readout()
        self.view_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        if self.topology_features != "wl":
            self._init_legacy_decoders()
        self._auxiliary = None
        self._diagnostics = {}

    def _init_legacy_decoders(self):
        self.text_decoder = nn.Linear(self.hidden_dim, self.in_feats)
        # Reconstruct only fixed observations, never the trainable field or maps.
        self.structure_decoder = nn.Linear(self.hidden_dim, self.structure_input_dim)
        # Asymmetric decoder preserves parent -> reply direction when available.
        self.edge_source = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_target = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(()))

    def _readout(self):
        return nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                             nn.GELU(), nn.LayerNorm(self.hidden_dim))

    @staticmethod
    def _coalesce_stances(edges, stance, n):
        """Deduplicate links with labels; unknown never overrides known evidence."""
        keep = edges[0] != edges[1]
        edges, stance = edges[:, keep], stance[keep]
        keys, inverse = torch.unique(edges[0] * n + edges[1], sorted=True, return_inverse=True)
        lower = torch.full_like(keys, 2).scatter_reduce(
            0, inverse, torch.where(stance < 0, 2, stance), reduce="amin", include_self=True)
        upper = torch.full_like(keys, -1).scatter_reduce(
            0, inverse, stance, reduce="amax", include_self=True)
        if bool(((lower < 2) & (lower != upper)).any()):
            raise ValueError("Duplicate propagation edges have conflicting stance labels")
        labels = torch.where(upper >= 0, upper, -1)
        return torch.stack((keys // n, keys % n)), labels

    @staticmethod
    def _rooted_edges(edges, stance, roots, n, directed):
        """Validate a forest, or recover an old undirected tree's root direction.

        A single linear CPU traversal avoids one GPU synchronization per depth.
        Isolates and directed components outside the event root are retained;
        their root depth is explicitly unknown. Unrooted undirected components
        with links cannot provide a trustworthy parent-relative orientation.
        """
        if not directed:
            edges = torch.stack((edges.amin(0), edges.amax(0)))
            edges, stance = DIGNN._coalesce_stances(edges, stance, n)
        pairs = edges.detach().cpu().t().tolist()
        root_ids = roots.detach().cpu().tolist()
        adjacency = [[] for _ in range(n)]
        indegree = [0] * n
        for parent, child in pairs:
            adjacency[parent].append(child)
            if directed:
                indegree[child] += 1
            else:
                adjacency[child].append(parent)
        if directed:
            if any(value > 1 for value in indegree) or any(indegree[root] for root in root_ids):
                raise ValueError("Energy mode requires a parent-to-child forest with root indegree zero")
            pending = deque(index for index, value in enumerate(indegree) if value == 0)
            remaining = indegree[:]
            visited = 0
            while pending:
                parent = pending.popleft()
                visited += 1
                for child in adjacency[parent]:
                    remaining[child] -= 1
                    if remaining[child] == 0:
                        pending.append(child)
            if visited != n:
                raise ValueError("Energy propagation edges cannot contain directed cycles")
        depth = [-1] * n
        pending = deque(root_ids)
        for root in root_ids:
            depth[root] = 0
        while pending:
            parent = pending.popleft()
            for child in adjacency[parent]:
                if depth[child] < 0:
                    depth[child] = depth[parent] + 1
                    pending.append(child)
        depth_tensor = torch.tensor(depth, device=edges.device, dtype=torch.long)
        if not directed:
            src_depth, dst_depth = depth_tensor[edges[0]], depth_tensor[edges[1]]
            if bool(((src_depth < 0) | (dst_depth < 0)).any()):
                raise ValueError("Cannot orient an unrooted old cache component; provide directed_edge_index")
            if bool(((src_depth - dst_depth).abs() != 1).any()):
                raise ValueError("Old energy cache must be a rooted tree, without same-depth links or cycles")
            flip = src_depth > dst_depth
            edges = torch.stack((torch.where(flip, edges[1], edges[0]),
                                 torch.where(flip, edges[0], edges[1])))
            if bool((torch.bincount(edges[1], minlength=n) > 1).any()):
                raise ValueError("Old energy cache must have a unique parent for each reply")
            edges, stance = DIGNN._coalesce_stances(edges, stance, n)
        return edges, stance, depth_tensor

    def _inputs(self, data):
        x = data.x.float()
        if x.ndim != 2 or x.size(0) == 0 or x.size(1) != self.in_feats:
            raise ValueError("x must be a nonempty [N, in_feats] tensor")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if batch.dtype != torch.long or batch.shape != (x.size(0),) or batch.device != x.device:
            raise ValueError("batch must be a long [N] tensor on the x device")
        if int(batch[0]) != 0 or bool((batch[1:] < batch[:-1]).any()):
            raise ValueError("DIGNN expects sorted PyG batch indices starting at zero")
        count = int(batch[-1]) + 1
        sizes = torch.bincount(batch, minlength=count)
        if bool((sizes == 0).any()):
            raise ValueError("Empty event graphs are unsupported")

        roots = getattr(data, "root_index", None)
        if roots is None:
            roots = getattr(data, "rootindex", None)
        if roots is None:
            roots = sizes.cumsum(0) - sizes
        else:
            roots = torch.as_tensor(roots, device=x.device).reshape(-1)
            if roots.dtype != torch.long:
                raise ValueError("root indices must have dtype torch.long")
        if (roots.numel() != count or bool((roots < 0).any())
                or bool((roots >= x.size(0)).any())
                or not torch.equal(batch[roots], torch.arange(count, device=x.device))):
            raise ValueError("Each event must have one globally indexed root in that event")

        edges = getattr(data, "directed_edge_index", None)
        directed = edges is not None
        stance = None
        if self.topology_features != "structural":
            stance = getattr(data, "directed_edge_stance", None) if directed else None
            # Pair old edge_stance with its own edge_index, never with a
            # differently ordered or differently sized directed edge tensor.
            if stance is None and getattr(data, "edge_stance", None) is not None:
                edges, stance = data.edge_index, data.edge_stance
                directed = False
            elif edges is None and getattr(data, "directed_edge_stance", None) is not None:
                raise ValueError("directed_edge_stance requires its matching directed_edge_index")
        if edges is None:
            edges = data.edge_index
        if (edges.ndim != 2 or edges.size(0) != 2 or edges.dtype != torch.long
                or edges.device != x.device):
            raise ValueError("edges must be a long [2, E] tensor on the x device")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= x.size(0)):
            raise ValueError("Edge endpoint is outside x")
        if bool((batch[edges[0]] != batch[edges[1]]).any()):
            raise ValueError("Edges cannot connect different events")
        if self.topology_features == "structural":
            # The structural ablation retains the original arbitrary-graph API.
            edges = coalesce(edges[:, edges[0] != edges[1]], num_nodes=x.size(0))
            return x, edges, batch, roots, sizes, None, None
        if stance is None:
            stance = edges.new_full((edges.size(1),), -1)
        else:
            stance = torch.as_tensor(stance, device=x.device).reshape(-1)
            if stance.dtype != torch.long or stance.numel() != edges.size(1):
                raise ValueError("Each propagation edge must have one aligned torch.long stance label")
            if not bool(((stance == -1) | (stance == 0) | (stance == 1)).all()):
                raise ValueError("Edge stance labels must be -1=unknown, 0=support or 1=deny")
        edges, stance = self._coalesce_stances(edges, stance, x.size(0))
        edges, stance, depth = self._rooted_edges(edges, stance, roots, x.size(0), directed)
        return x, edges, batch, roots, sizes, stance, depth

    @staticmethod
    def _structural_features(edges, batch, roots, sizes, dtype):
        n = batch.numel()
        in_degree = torch.bincount(edges[1], minlength=n).to(dtype)
        out_degree = torch.bincount(edges[0], minlength=n).to(dtype)
        is_root = torch.zeros(n, device=batch.device, dtype=dtype)
        is_root[roots] = 1
        return torch.stack((in_degree.log1p(), out_degree.log1p(), is_root,
                            (in_degree == 0).to(dtype), (out_degree == 0).to(dtype),
                            sizes[batch].to(dtype).log1p()), dim=-1)

    @classmethod
    def _energy_descriptors(cls, edges, stance, batch, roots, sizes, depth, dtype):
        n = batch.numel()
        outgoing = torch.bincount(edges[0], minlength=n).to(dtype)
        support = torch.bincount(edges[0, stance == 0], minlength=n).to(dtype)
        deny = torch.bincount(edges[0, stance == 1], minlength=n).to(dtype)
        unknown = torch.bincount(edges[0, stance < 0], minlength=n).to(dtype)
        source = torch.stack((support.log1p(), deny.log1p()), dim=-1)
        role = torch.full((n,), 3, device=batch.device, dtype=torch.long)
        role[edges[1]] = torch.where(stance >= 0, stance + 1, 3)
        role[roots] = 0
        coverage = (support + deny) / outgoing.clamp_min(1)
        descriptors = torch.cat((
            cls._structural_features(edges, batch, roots, sizes, dtype), source,
            F.one_hot(role, num_classes=4).to(dtype),
            coverage[:, None], unknown.log1p()[:, None],
            depth.clamp_min(0).to(dtype).log1p()[:, None], (depth >= 0).to(dtype)[:, None]), dim=-1)
        return descriptors, source

    def encode_views(self, data):
        """Expose separate views, fixed descriptors, and differentiable fields."""
        x, edges, batch, roots, sizes, stance, depth = self._inputs(data)
        if self.topology_features == "wl":
            views = self.topology_encoder(edges, stance, batch, roots)
            text = self.text_encoder(x)
            text_graph = self.text_readout(torch.cat(
                (text[roots], global_mean_pool(text, batch, size=sizes.numel())), dim=-1))
            return dict(text_nodes=text, text_graph=text_graph, x=x, edge_index=edges,
                        edge_stance=stance, batch=batch, graph_count=sizes.numel(), **views)
        energy_views = {}
        if self.energy_field is None:
            descriptors = self._structural_features(edges, batch, roots, sizes, x.dtype)
            topology_input = descriptors
        else:
            descriptors, source = self._energy_descriptors(
                edges, stance, batch, roots, sizes, depth, x.dtype)
            energy = self.energy_field(source, edges, stance, batch)
            topology_input = torch.cat((descriptors, energy["fields"].flatten(1),
                                        energy["residuals"].flatten(1), energy["source_energy"],
                                        energy["relation_energy"].flatten(1)), dim=-1)
            energy_views = {"energy_source": source, "energy_fields": energy["fields"],
                            "energy_residuals": energy["residuals"],
                            "energy_source_energy": energy["source_energy"],
                            "energy_relation_energy": energy["relation_energy"],
                            "energy_initial_energy": energy["initial_energy"],
                            "energy_final_energy": energy["final_energy"]}
        topology = self.topology_encoder(topology_input, to_undirected(edges, num_nodes=x.size(0)))
        text = self.text_encoder(x)
        count = sizes.numel()
        topology_graph = self.topology_readout(torch.cat(
            (topology[roots], global_mean_pool(topology, batch, size=count)), dim=-1))
        text_graph = self.text_readout(torch.cat(
            (text[roots], global_mean_pool(text, batch, size=count)), dim=-1))
        return dict(topology_nodes=topology, text_nodes=text,
                    topology_graph=topology_graph, text_graph=text_graph,
                    structure_features=descriptors, x=x, edge_index=edges,
                    edge_stance=stance, batch=batch, graph_count=count, **energy_views)

    @staticmethod
    def _event_mse(prediction, target, batch, count):
        per_node = (prediction - target.detach()).square().mean(-1, keepdim=True)
        return global_mean_pool(per_node, batch, size=count).mean()

    def _edge_reconstruction(self, topology, edges, batch, count):
        if edges.size(1) == 0:
            return topology.sum() * 0.0
        # Coalesced edges are sorted by source, as required by PyG's sampler.
        # Negatives are within events, exclude positives/self-loops, and remain
        # sparse. Complete events simply have no available negative pairs.
        negatives = batched_negative_sampling(edges, batch, method="sparse")
        source = self.edge_source(topology)
        target = self.edge_target(topology)
        event_loss = topology.new_zeros(count)
        available_terms = topology.new_zeros(count)
        for pairs, positive in ((edges, True), (negatives, False)):
            if pairs.size(1) == 0:
                continue
            logits = (source[pairs[0]] * target[pairs[1]]).sum(-1) / math.sqrt(self.hidden_dim)
            logits = logits + self.edge_bias
            labels = torch.full_like(logits, float(positive))
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            event_ids = batch[pairs[0]]
            pair_counts = torch.bincount(event_ids, minlength=count).to(losses.dtype)
            event_loss = event_loss + losses.new_zeros(count).index_add(0, event_ids, losses) / pair_counts.clamp_min(1)
            available_terms = available_terms + (pair_counts > 0).to(losses.dtype)
        # Equal event weights and balanced positive/negative terms per event.
        return (event_loss / available_terms.clamp_min(1)).mean()

    def forward(self, data):
        self._auxiliary = None
        self._diagnostics = {}
        views = self.encode_views(data)
        topology, text = views["topology_graph"], views["text_graph"]
        stacked = torch.stack((topology, text), dim=1)
        if self.fusion == "attention":
            attention = self.view_attention(stacked).squeeze(-1).softmax(-1)
        else:
            weights = {"mean": (0.5, 0.5), "topology": (1.0, 0.0), "text": (0.0, 1.0)}[self.fusion]
            attention = stacked.new_tensor(weights).expand(topology.size(0), -1)
        representation = (stacked * attention.unsqueeze(-1)).sum(1)
        output = F.log_softmax(self.classifier(representation), dim=-1)
        self._diagnostics["view_attention"] = attention.detach()
        if self.topology_features == "wl":
            self._diagnostics.update(views["wl_diagnostics"])
        if self.energy_field is not None:
            event_edges = torch.bincount(views["batch"][views["edge_index"][0]], minlength=views["graph_count"])
            known_edges = torch.bincount(
                views["batch"][views["edge_index"][0, views["edge_stance"] >= 0]],
                minlength=views["graph_count"])
            self._diagnostics.update(
                energy_initial_energy=views["energy_initial_energy"].detach(),
                energy_final_energy=views["energy_final_energy"].detach(),
                energy_label_coverage=(known_edges.to(output.dtype) / event_edges.clamp_min(1)).detach())

        if self.training:
            zero = representation.sum() * 0.0
            losses = {name: zero for name in self.loss_weights}
            batch, count = views["batch"], views["graph_count"]
            if self.loss_weights["text_reconstruction"] > 0:
                losses["text_reconstruction"] = self._event_mse(
                    self.text_decoder(views["text_nodes"]), views["x"], batch, count)
            if self.loss_weights["structure_reconstruction"] > 0:
                losses["structure_reconstruction"] = self._event_mse(
                    self.structure_decoder(views["topology_nodes"]), views["structure_features"], batch, count)
            if self.loss_weights["edge_reconstruction"] > 0:
                losses["edge_reconstruction"] = self._edge_reconstruction(
                    views["topology_nodes"], views["edge_index"], batch, count)
            if self.loss_weights["independence"] > 0:
                losses["independence"] = normalized_hsic(topology, text)
            self._auxiliary = sum(self.loss_weights[name] * value for name, value in losses.items())
            self._diagnostics.update({name: value.detach() for name, value in losses.items()})
        dummy = output.new_zeros(output.size(0), self.max_hop, 1)
        return output, dummy, dummy, dummy

    def auxiliary_loss(self):
        if not self.training or self._auxiliary is None:
            return self.classifier.weight.new_zeros(())
        return self._auxiliary

    def get_diagnostics(self):
        """Detached attention [B, 2] (topology, text) and raw training losses."""
        return dict(self._diagnostics)

    def classification_loss(self, output, target):
        return F.nll_loss(output, target.reshape(-1).long())

    def physics_loss(self, U, S, D, true_state):
        return self.classifier.weight.new_zeros(())

    def compute_loss(self, data):
        output = self(data)[0]
        return self.classification_loss(output, data.y) + self.auxiliary_loss()

    def init_optimizer(self, args=None):
        args = self.args if args is None else args
        return torch.optim.Adam(self.parameters(), lr=float(getattr(args, "lr", 0.0005)),
                                weight_decay=float(getattr(args, "weight_decay", 0.0001)))
