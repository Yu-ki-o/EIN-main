import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class MLPSemanticChangeEncoder(nn.Module):
    """
    Encodes node-level semantic change between two aligned graph views.

    The direction convention is always:

        delta = deny_nodes - support_nodes

    Only the signed change and its absolute magnitude are encoded. Edge or
    node uncertainty is intentionally kept outside this module so uncertainty
    routing/trend modeling can remain an independent, replaceable branch.
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = (
            self.input_dim if output_dim is None else int(output_dim)
        )
        mlp_hidden_dim = (
            self.input_dim if hidden_dim is None else int(hidden_dim)
        )

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if mlp_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        # Bias-free layers guarantee that identical support/deny views produce
        # an exact zero change representation.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, mlp_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden_dim, self.output_dim, bias=False),
        )

    def change_features(self, support_nodes, deny_nodes):
        self._validate_inputs(support_nodes, deny_nodes)
        delta = deny_nodes - support_nodes
        return torch.cat((delta, delta.abs()), dim=-1)

    def forward(self, support_nodes, deny_nodes, **kwargs):
        features = self.change_features(support_nodes, deny_nodes)
        return self.encoder(features)

    def _validate_inputs(self, support_nodes, deny_nodes):
        if support_nodes.shape != deny_nodes.shape:
            raise ValueError(
                "support_nodes and deny_nodes must have identical shapes, "
                "got {} and {}".format(
                    tuple(support_nodes.shape),
                    tuple(deny_nodes.shape),
                )
            )
        if support_nodes.size(-1) != self.input_dim:
            raise ValueError(
                "expected node feature dimension {}, got {}".format(
                    self.input_dim,
                    support_nodes.size(-1),
                )
            )


class _DPGAAlignmentLayer(nn.Module):
    """
    Pseudo-node alignment layer inspired by dynamic pathway-based graph
    alignment. It exchanges information through a small pseudo-node bottleneck
    instead of fully connecting the two graph views.
    """

    def __init__(self, hidden_dim, dropout=0.0, temperature=1.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temperature = max(1e-3, float(temperature))
        self.dropout = float(dropout)

        self.edge_key = nn.Embedding(4, self.hidden_dim)
        self.edge_value = nn.Embedding(4, self.hidden_dim)

        self.pseudo_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.node_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.pseudo_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.pseudo_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.pseudo_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pseudo_norm = nn.LayerNorm(self.hidden_dim)
        self.node_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        support_nodes,
        deny_nodes,
        pseudo_nodes,
        node_mask,
        support_weight=None,
        deny_weight=None,
    ):
        pseudo_nodes = self._update_pseudo_nodes(
            support_nodes,
            deny_nodes,
            pseudo_nodes,
            node_mask,
            support_weight=support_weight,
            deny_weight=deny_weight,
        )
        support_nodes = self._update_view_nodes(
            support_nodes,
            pseudo_nodes,
            node_mask,
            edge_type=2,
        )
        deny_nodes = self._update_view_nodes(
            deny_nodes,
            pseudo_nodes,
            node_mask,
            edge_type=3,
        )
        return support_nodes, deny_nodes, pseudo_nodes

    def _update_pseudo_nodes(
        self,
        support_nodes,
        deny_nodes,
        pseudo_nodes,
        node_mask,
        support_weight=None,
        deny_weight=None,
    ):
        _, max_nodes, _ = support_nodes.shape
        nodes = torch.cat((support_nodes, deny_nodes), dim=1)
        mask = torch.cat((node_mask, node_mask), dim=1)
        edge_type = torch.cat(
            (
                torch.zeros(max_nodes, dtype=torch.long, device=nodes.device),
                torch.ones(max_nodes, dtype=torch.long, device=nodes.device),
            ),
            dim=0,
        )
        edge_key = self.edge_key(edge_type).view(1, 1, max_nodes * 2, -1)
        edge_value = self.edge_value(edge_type).view(1, max_nodes * 2, -1)

        query = self.pseudo_query(pseudo_nodes).unsqueeze(2)
        key = self.node_key(nodes).unsqueeze(1) + edge_key
        score = (query * key).sum(dim=-1)
        score = score / (self.hidden_dim ** 0.5 * self.temperature)
        score = score.masked_fill(~mask.unsqueeze(1), -1e9)

        if support_weight is not None and deny_weight is not None:
            weight = torch.cat((support_weight, deny_weight), dim=1)
            score = score + torch.log(weight.clamp_min(1e-6)).unsqueeze(1)

        attention = F.softmax(score, dim=-1)
        value = self.node_value(nodes) + edge_value
        context = torch.matmul(attention, value)
        update = self.pseudo_update(torch.cat((pseudo_nodes, context), dim=-1))
        update = F.dropout(update, p=self.dropout, training=self.training)
        return self.pseudo_norm(pseudo_nodes + update)

    def _update_view_nodes(
        self,
        view_nodes,
        pseudo_nodes,
        node_mask,
        edge_type,
    ):
        edge_ids = torch.full(
            (pseudo_nodes.size(1),),
            int(edge_type),
            dtype=torch.long,
            device=view_nodes.device,
        )
        edge_key = self.edge_key(edge_ids).view(1, 1, pseudo_nodes.size(1), -1)
        edge_value = self.edge_value(edge_ids).view(1, pseudo_nodes.size(1), -1)

        query = self.node_query(view_nodes).unsqueeze(2)
        key = self.pseudo_key(pseudo_nodes).unsqueeze(1) + edge_key
        score = (query * key).sum(dim=-1)
        score = score / (self.hidden_dim ** 0.5 * self.temperature)
        attention = F.softmax(score, dim=-1)

        value = self.pseudo_value(pseudo_nodes) + edge_value
        context = torch.matmul(attention, value)
        update = self.node_update(torch.cat((view_nodes, context), dim=-1))
        update = F.dropout(update, p=self.dropout, training=self.training)
        updated = self.node_norm(view_nodes + update)
        return torch.where(node_mask.unsqueeze(-1), updated, view_nodes)


class DPGASemanticChangeEncoder(nn.Module):
    """
    DPGA-style semantic change encoder for aligned support/deny views.

    The encoder keeps the existing signed-delta change signal, then modulates
    it with pseudo-node alignment context. This preserves the useful invariant
    that identical support/deny node features produce exact zero change while
    still allowing graph-level dynamic pathways to decide which changes matter.
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        dropout=0.0,
        pseudo_nodes=4,
        layers=1,
        attention_temperature=1.0,
        modulation_scale=0.5,
        use_node_weights=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = (
            self.input_dim if output_dim is None else int(output_dim)
        )
        self.hidden_dim = (
            self.input_dim if hidden_dim is None else int(hidden_dim)
        )
        self.num_pseudo_nodes = int(pseudo_nodes)
        self.num_layers = int(layers)
        self.dropout = float(dropout)
        self.modulation_scale = max(0.0, float(modulation_scale))
        self.use_node_weights = bool(use_node_weights)

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_pseudo_nodes <= 0:
            raise ValueError("pseudo_nodes must be positive")
        if self.num_layers <= 0:
            raise ValueError("layers must be positive")

        self.local_change = MLPSemanticChangeEncoder(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.node_projection = nn.Linear(
            self.input_dim,
            self.hidden_dim,
            bias=False,
        )
        self.pseudo_nodes = nn.Parameter(
            torch.empty(self.num_pseudo_nodes, self.hidden_dim)
        )
        self.layers = nn.ModuleList(
            [
                _DPGAAlignmentLayer(
                    self.hidden_dim,
                    dropout=self.dropout,
                    temperature=attention_temperature,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.context_projection = nn.Linear(
            self.hidden_dim,
            self.output_dim,
            bias=False,
        )
        self.graph_projection = nn.Linear(
            self.hidden_dim,
            self.output_dim,
            bias=False,
        )
        self.modulator = nn.Sequential(
            nn.Linear(self.output_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.pseudo_nodes)

    def forward(
        self,
        support_nodes,
        deny_nodes,
        batch=None,
        support_node_weight=None,
        deny_node_weight=None,
        node_keep=None,
        **kwargs,
    ):
        self.local_change._validate_inputs(support_nodes, deny_nodes)
        if support_nodes.size(0) == 0:
            return support_nodes.new_zeros((0, self.output_dim))

        local_change = self.local_change(support_nodes, deny_nodes)
        batch = self._batch_or_default(batch, support_nodes)

        support_dense, node_mask = to_dense_batch(
            self.node_projection(support_nodes),
            batch,
        )
        deny_dense, deny_mask = to_dense_batch(
            self.node_projection(deny_nodes),
            batch,
        )
        node_mask = node_mask & deny_mask

        support_weight, deny_weight = self._dense_view_weights(
            support_nodes,
            batch,
            support_node_weight,
            deny_node_weight,
            node_keep,
        )
        pseudo_dense = self.pseudo_nodes.unsqueeze(0).expand(
            support_dense.size(0),
            -1,
            -1,
        )
        for layer in self.layers:
            support_dense, deny_dense, pseudo_dense = layer(
                support_dense,
                deny_dense,
                pseudo_dense,
                node_mask,
                support_weight=support_weight,
                deny_weight=deny_weight,
            )

        context_delta = self.context_projection(
            deny_dense[node_mask] - support_dense[node_mask]
        )
        graph_context = self.graph_projection(pseudo_dense.mean(dim=1))
        graph_context = graph_context[batch.long()]

        modulation_features = torch.cat(
            (
                context_delta,
                context_delta.abs(),
                graph_context,
            ),
            dim=-1,
        )
        modulation = torch.tanh(self.modulator(modulation_features))
        scale = 1.0 + self.modulation_scale * modulation
        return local_change * scale

    def change_features(self, support_nodes, deny_nodes):
        return self.local_change.change_features(support_nodes, deny_nodes)

    def _batch_or_default(self, batch, support_nodes):
        if batch is None:
            return torch.zeros(
                support_nodes.size(0),
                dtype=torch.long,
                device=support_nodes.device,
            )
        return batch.to(device=support_nodes.device, dtype=torch.long)

    def _dense_view_weights(
        self,
        support_nodes,
        batch,
        support_node_weight,
        deny_node_weight,
        node_keep,
    ):
        if not self.use_node_weights:
            return None, None

        support_weight = self._weight_or_ones(support_node_weight, support_nodes)
        deny_weight = self._weight_or_ones(deny_node_weight, support_nodes)
        if node_keep is not None:
            keep = node_keep.to(
                device=support_nodes.device,
                dtype=support_nodes.dtype,
            ).view(-1)
            support_weight = support_weight * keep
            deny_weight = deny_weight * keep

        support_dense, _ = to_dense_batch(
            support_weight.view(-1, 1),
            batch,
        )
        deny_dense, _ = to_dense_batch(
            deny_weight.view(-1, 1),
            batch,
        )
        return support_dense.squeeze(-1), deny_dense.squeeze(-1)

    def _weight_or_ones(self, weight, support_nodes):
        if weight is None:
            return support_nodes.new_ones(support_nodes.size(0))
        return weight.to(
            device=support_nodes.device,
            dtype=support_nodes.dtype,
        ).view(-1)


def build_semantic_change_encoder(
    encoder_name,
    input_dim,
    output_dim=None,
    hidden_dim=None,
    dropout=0.0,
    args=None,
):
    """
    Factory kept at the model boundary so the MLP can later be replaced
    without changing the caller's forward interface.
    """

    name = str(encoder_name).strip().lower()
    if name == "mlp":
        return MLPSemanticChangeEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if name == "dpga":
        return DPGASemanticChangeEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pseudo_nodes=int(getattr(args, "dpga_pseudo_nodes", 4)),
            layers=int(getattr(args, "dpga_layers", 1)),
            attention_temperature=float(
                getattr(args, "dpga_attention_temperature", 1.0)
            ),
            modulation_scale=float(
                getattr(args, "dpga_modulation_scale", 0.5)
            ),
            use_node_weights=bool(
                getattr(args, "dpga_use_node_weights", True)
            ),
        )
    raise ValueError(
        "unsupported semantic change encoder: {!r}".format(encoder_name)
    )
