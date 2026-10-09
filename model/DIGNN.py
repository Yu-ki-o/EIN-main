"""DIGNN-inspired, inductive graph classifier for rumor detection.

Topology and text are encoded separately. Unlike the paper's transductive
MLP(A), a structure-only GIN handles unseen, variable-size propagation trees.
Reconstruction retains each view; normalized RBF HSIC is an independence
surrogate, NOT the variational mutual-information bound in paper Eq. (8).
See docs/DIGNN.md for the adaptation and loss definitions.
"""

from __future__ import annotations

import math

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


class StructureEncoder(nn.Module):
    """Rooted GIN on structural descriptors, with no text input."""

    def __init__(self, hidden_dim, num_layers, dropout):
        super().__init__()
        self.projection = nn.Linear(6, hidden_dim)
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


class DIGNN(nn.Module):
    """Return (log_probs, U, S, D) for EINTrainer.

    Inputs: x [N, in_feats], edge_index [2, E], optional directed_edge_index,
    PyG batch, and optional globally indexed root_index/rootindex [B]. In the
    project's cache the source is the first node of each event. Labels,
    stances, user_state and timestamps are never read by forward().
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
        self.loss_weights = {
            "text_reconstruction": float(getattr(args, "dignn_text_recon_weight", 0.1)),
            "structure_reconstruction": float(getattr(args, "dignn_structure_recon_weight", 0.1)),
            "edge_reconstruction": float(getattr(args, "dignn_edge_recon_weight", 0.1)),
            "independence": float(getattr(args, "dignn_independence_weight", 0.01)),
        }
        if min(self.in_feats, self.hidden_dim, num_classes, self.max_hop, layers) < 1:
            raise ValueError("dimensions, classes, max_hop and layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.fusion not in {"attention", "mean", "topology", "text"}:
            raise ValueError("dignn_fusion must be attention, mean, topology or text")
        if any(not math.isfinite(w) or w < 0 for w in self.loss_weights.values()):
            raise ValueError("DIGNN loss weights must be finite and nonnegative")

        self.topology_encoder = StructureEncoder(self.hidden_dim, layers, dropout)
        self.text_encoder = nn.Sequential(
            nn.Linear(self.in_feats, self.hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        # Root semantics and whole-event content are retained within each view.
        self.topology_readout = self._readout()
        self.text_readout = self._readout()
        self.view_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.text_decoder = nn.Linear(self.hidden_dim, self.in_feats)
        self.structure_decoder = nn.Linear(self.hidden_dim, 6)
        # Asymmetric decoder preserves parent -> reply direction when available.
        self.edge_source = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_target = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(()))
        self._auxiliary = None
        self._diagnostics = {}

    def _readout(self):
        return nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                             nn.GELU(), nn.LayerNorm(self.hidden_dim))

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
        if edges is None:
            edges = data.edge_index
        if (edges.ndim != 2 or edges.size(0) != 2 or edges.dtype != torch.long
                or edges.device != x.device):
            raise ValueError("edges must be a long [2, E] tensor on the x device")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= x.size(0)):
            raise ValueError("Edge endpoint is outside x")
        if bool((batch[edges[0]] != batch[edges[1]]).any()):
            raise ValueError("Edges cannot connect different events")
        # Duplicate links and self-loops are not additional propagation evidence.
        edges = coalesce(edges[:, edges[0] != edges[1]], num_nodes=x.size(0))
        return x, edges, batch, roots, sizes

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

    def encode_views(self, data):
        """Expose independent node/event representations for analysis."""
        x, edges, batch, roots, sizes = self._inputs(data)
        descriptors = self._structural_features(edges, batch, roots, sizes, x.dtype)
        topology = self.topology_encoder(descriptors, to_undirected(edges, num_nodes=x.size(0)))
        text = self.text_encoder(x)
        count = sizes.numel()
        topology_graph = self.topology_readout(torch.cat(
            (topology[roots], global_mean_pool(topology, batch, size=count)), dim=-1))
        text_graph = self.text_readout(torch.cat(
            (text[roots], global_mean_pool(text, batch, size=count)), dim=-1))
        return dict(topology_nodes=topology, text_nodes=text,
                    topology_graph=topology_graph, text_graph=text_graph,
                    structure_features=descriptors, x=x, edge_index=edges,
                    batch=batch, graph_count=count)

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
