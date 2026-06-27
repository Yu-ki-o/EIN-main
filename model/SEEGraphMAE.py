import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.utils import to_undirected


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        num_layers = max(1, int(num_layers))
        if num_layers == 1:
            self.layers = nn.Sequential(nn.Linear(in_dim, out_dim))
            return

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEEGraphMAE(nn.Module):
    """
    SEmantic Evolving Graph Masked AutoEncoder for rumor detection.

    The implementation follows the paper's three semantic evolving branches:
    parent-to-child local reconstruction, child-to-parent local reconstruction,
    and masked global graph reconstruction. The graph representation is the
    mean-pooled concatenation of the three encoders.
    """

    def __init__(
        self,
        in_feats,
        hid_feats,
        num_classes,
        args,
        device,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.in_feats = int(in_feats)
        self.hidden_dim = int(hid_feats)
        self.num_classes = int(num_classes)
        self.dropout = float(getattr(args, "see_dropout", 0.0))

        self.mask_ratio = float(getattr(args, "see_mask_ratio", 0.25))
        self.alpha_rec = float(getattr(args, "see_alpha_rec", 0.1))
        self.alpha_uni = float(getattr(args, "see_alpha_uni", 0.5))
        self.uniformity_t = float(getattr(args, "see_uniformity_t", 2.0))
        self.encoder_layers = int(getattr(args, "see_mlp_layers", 2))
        self.decoder_layers = int(getattr(args, "see_decoder_layers", 2))

        pool_name = str(getattr(args, "see_global_pool", "mean")).lower()
        self.pool_is_sum = "sum" in pool_name
        self.global_pool = global_add_pool if self.pool_is_sum else global_mean_pool

        self.top_down_encoder = MLP(
            self.in_feats,
            self.hidden_dim,
            self.hidden_dim,
            num_layers=self.encoder_layers,
            dropout=self.dropout,
        )
        self.bottom_up_encoder = MLP(
            self.in_feats,
            self.hidden_dim,
            self.hidden_dim,
            num_layers=self.encoder_layers,
            dropout=self.dropout,
        )
        self.top_down_decoder = MLP(
            self.hidden_dim,
            self.hidden_dim,
            self.in_feats,
            num_layers=self.decoder_layers,
            dropout=self.dropout,
        )
        self.bottom_up_decoder = MLP(
            self.hidden_dim,
            self.hidden_dim,
            self.in_feats,
            num_layers=self.decoder_layers,
            dropout=self.dropout,
        )

        self.global_encoder = nn.ModuleList(
            [
                GCNConv(self.in_feats, self.hidden_dim),
                GCNConv(self.hidden_dim, self.hidden_dim),
            ]
        )
        self.global_decoder = GCNConv(self.hidden_dim, self.in_feats)
        self.mask_token = nn.Parameter(torch.zeros(self.in_feats))

        classifier_hidden = int(getattr(args, "see_classifier_hidden_dim", 0))
        classifier_in = self.hidden_dim * 3
        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(classifier_hidden, self.num_classes),
            )
        else:
            self.classifier = nn.Linear(classifier_in, self.num_classes)

        class_weights = getattr(args, "classification_class_weights", None)
        if class_weights is None:
            self.register_buffer("classification_class_weights", torch.empty(0))
        else:
            self.register_buffer(
                "classification_class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def self_supervised_parameters(self):
        for module in (
            self.top_down_encoder,
            self.bottom_up_encoder,
            self.top_down_decoder,
            self.bottom_up_decoder,
            self.global_encoder,
            self.global_decoder,
        ):
            yield from module.parameters()
        yield self.mask_token

    def classification_loss(self, output, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.nll_loss(output, target.view(-1).long(), weight=weight)

    def physics_loss(self, U, S, D, true_state):
        return self._zero()

    def _zero(self):
        return self.mask_token.new_zeros(())

    def _batch_vector(self, data):
        if hasattr(data, "batch"):
            return data.batch.long()
        return torch.zeros(
            data.x.size(0),
            dtype=torch.long,
            device=data.x.device,
        )

    def _directed_edge_index(self, data):
        edge_index = getattr(data, "directed_edge_index", None)
        if edge_index is None:
            edge_index = data.edge_index
        return edge_index.long()

    def _undirected_edge_index(self, data):
        return to_undirected(
            self._directed_edge_index(data),
            num_nodes=data.x.size(0),
        )

    def _pool(self, node_hidden, batch):
        return self.global_pool(node_hidden, batch)

    def _encode_global_nodes(self, x, edge_index):
        hidden = x
        for conv in self.global_encoder:
            hidden = conv(hidden, edge_index)
            hidden = F.relu(hidden)
            hidden = F.dropout(
                hidden,
                p=self.dropout,
                training=self.training,
            )
        return hidden

    def encode_graph(self, data, node_mask=None):
        x = data.x.float()
        if node_mask is not None:
            x = x * node_mask.view(-1, 1).to(dtype=x.dtype)

        batch = self._batch_vector(data)
        global_edge_index = self._undirected_edge_index(data)

        top_down_nodes = self.top_down_encoder(x)
        bottom_up_nodes = self.bottom_up_encoder(x)
        global_nodes = self._encode_global_nodes(x, global_edge_index)

        top_down_graph = self._pool(top_down_nodes, batch)
        bottom_up_graph = self._pool(bottom_up_nodes, batch)
        global_graph = self._pool(global_nodes, batch)
        return torch.cat(
            (top_down_graph, bottom_up_graph, global_graph),
            dim=-1,
        )

    def predict_logits(self, data, node_mask=None):
        graph_repr = self.encode_graph(data, node_mask=node_mask)
        return self.classifier(graph_repr)

    def forward(self, data, node_mask=None, return_logits=False):
        logits = self.predict_logits(data, node_mask=node_mask)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def local_reconstruction_losses(self, data):
        edge_index = self._directed_edge_index(data)
        if edge_index.numel() == 0:
            return self._zero(), self._zero()

        src, dst = edge_index
        x = data.x.float()
        parent_x = x[src]
        child_x = x[dst]

        reconstructed_child = self.top_down_decoder(
            self.top_down_encoder(parent_x)
        )
        reconstructed_parent = self.bottom_up_decoder(
            self.bottom_up_encoder(child_x)
        )
        rec_top_down = F.mse_loss(reconstructed_child, child_x)
        rec_bottom_up = F.mse_loss(reconstructed_parent, parent_x)
        return rec_top_down, rec_bottom_up

    def _sample_mask_nodes(self, data):
        x = data.x
        batch = self._batch_vector(data)
        node_mask = torch.zeros(
            x.size(0),
            dtype=torch.bool,
            device=x.device,
        )
        if x.size(0) == 0:
            return node_mask

        num_graphs = int(batch.max().item()) + 1
        for graph_id in range(num_graphs):
            graph_nodes = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            if graph_nodes.numel() == 0:
                continue
            mask_count = int(math.ceil(graph_nodes.numel() * self.mask_ratio))
            mask_count = min(max(mask_count, 1), graph_nodes.numel())
            chosen = torch.randperm(
                graph_nodes.numel(),
                device=x.device,
            )[:mask_count]
            node_mask[graph_nodes[chosen]] = True
        return node_mask

    def global_reconstruction_loss(self, data):
        x = data.x.float()
        mask_nodes = self._sample_mask_nodes(data)
        if not mask_nodes.any():
            return self._zero()

        masked_x = x.clone()
        mask_value = self.mask_token.to(dtype=x.dtype).view(1, -1).expand(
            int(mask_nodes.sum().item()),
            -1,
        )
        masked_x[mask_nodes] = mask_value
        edge_index = self._undirected_edge_index(data)
        hidden = self._encode_global_nodes(masked_x, edge_index)
        reconstructed = self.global_decoder(hidden, edge_index)
        return F.mse_loss(reconstructed[mask_nodes], x[mask_nodes])

    def uniformity_loss(self, graph_repr):
        if graph_repr.size(0) <= 1:
            return self._zero()
        graph_repr = F.normalize(graph_repr, p=2, dim=-1)
        distances = torch.pdist(graph_repr, p=2).pow(2)
        if distances.numel() == 0:
            return self._zero()
        potential = torch.exp(-self.uniformity_t * distances)
        return torch.log(potential.mean().clamp_min(1e-12))

    def self_supervised_loss(self, data, include_uniformity=True):
        rec_top_down, rec_bottom_up = self.local_reconstruction_losses(data)
        rec_global = self.global_reconstruction_loss(data)

        if include_uniformity:
            graph_repr = self.encode_graph(data)
            uniformity = self.uniformity_loss(graph_repr)
        else:
            uniformity = self._zero()

        rec_total = rec_top_down + rec_bottom_up + rec_global
        loss = self.alpha_rec * rec_total + self.alpha_uni * uniformity
        metrics = {
            "rec_top_down": rec_top_down.detach(),
            "rec_bottom_up": rec_bottom_up.detach(),
            "rec_global": rec_global.detach(),
            "uniformity": uniformity.detach(),
            "ssl_loss": loss.detach(),
        }
        return loss, metrics

    def __repr__(self):
        return self.__class__.__name__
