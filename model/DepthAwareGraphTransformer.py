import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch


class DepthAwareTransformerBlock(nn.Module):
    """Transformer block whose attention logits include relative-depth bias."""

    def __init__(
        self,
        hidden_dim,
        heads,
        feedforward_dim,
        dropout,
        max_depth,
        use_relative_depth_bias=True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.max_depth = int(max_depth)
        self.use_relative_depth_bias = bool(use_relative_depth_bias)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

        if self.use_relative_depth_bias:
            self.relative_depth_bias = nn.Embedding(
                self.max_depth * 2 + 1,
                self.heads,
            )
        else:
            self.relative_depth_bias = None

        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, self.hidden_dim),
            nn.Dropout(dropout),
        )

    def _relative_depth_indices(self, depth):
        relative_depth = depth.unsqueeze(2) - depth.unsqueeze(1)
        return relative_depth.clamp(-self.max_depth, self.max_depth).long() + (
            self.max_depth
        )

    def forward(self, node_hidden, valid_mask, depth):
        batch_size, max_nodes, _ = node_hidden.size()
        residual = node_hidden
        hidden = self.norm1(node_hidden)

        qkv = self.qkv(hidden).view(
            batch_size,
            max_nodes,
            3,
            self.heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention_score = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * self.scale

        if self.relative_depth_bias is not None:
            depth_bias = self.relative_depth_bias(
                self._relative_depth_indices(depth)
            )
            depth_bias = depth_bias.permute(0, 3, 1, 2)
            attention_score = attention_score + depth_bias

        key_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        attention_score = attention_score.masked_fill(
            ~key_mask,
            torch.finfo(attention_score.dtype).min,
        )
        attention = F.softmax(attention_score, dim=-1)
        attention = attention.masked_fill(~key_mask, 0.0)
        attention = self.attention_dropout(attention)

        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            max_nodes,
            self.hidden_dim,
        )
        context = self.output_dropout(self.output_projection(context))
        node_hidden = residual + context
        node_hidden = node_hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        node_hidden = node_hidden + self.ffn(self.norm2(node_hidden))
        return node_hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0), attention


class DepthAwareGraphTransformer(nn.Module):
    """
    Clean graph-level Transformer for rumor classification.

    Node features are fused with root-relative depth embeddings before every
    node enters self-attention. Each attention block can also add a learnable
    relative-depth bias directly to the attention logits.
    """

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
        self.in_feats = int(in_feats)
        self.hidden_dim = int(hid_feats)
        self.num_classes = int(num_classes)
        self.max_hop = int(getattr(args, "max_hop", 32))
        self.max_depth = max(
            0,
            int(getattr(args, "depth_transformer_max_depth", self.max_hop)),
        )
        self.depth_dim = max(
            1,
            int(getattr(args, "depth_transformer_depth_dim", self.hidden_dim)),
        )
        self.dropout = float(
            getattr(args, "depth_transformer_dropout", getattr(args, "dropout", 0.0))
        )
        self.heads = max(
            1,
            int(getattr(args, "depth_transformer_heads", 4)),
        )
        if self.hidden_dim % self.heads != 0:
            raise ValueError(
                "hidden_dim {} must be divisible by "
                "depth_transformer_heads {}".format(
                    self.hidden_dim,
                    self.heads,
                )
            )
        self.layers = max(
            1,
            int(getattr(args, "depth_transformer_layers", 1)),
        )
        self.feedforward_dim = max(
            self.hidden_dim,
            int(
                getattr(
                    args,
                    "depth_transformer_ffn_dim",
                    self.hidden_dim * 2,
                )
            ),
        )
        self.pool = str(
            getattr(args, "depth_transformer_pool", getattr(args, "global_pool", "mean"))
        ).strip().lower()
        if self.pool not in {"mean", "sum", "root"}:
            raise ValueError(
                "depth_transformer_pool must be one of "
                "['mean', 'root', 'sum'], got {}".format(self.pool)
            )
        self.use_root_context = bool(
            getattr(args, "depth_transformer_use_root_context", True)
        )
        self.use_relative_depth_bias = bool(
            getattr(args, "depth_transformer_use_relative_depth_bias", True)
        )
        self.log_attention = bool(
            getattr(args, "depth_transformer_log_attention", False)
        )

        self.node_projection = nn.Sequential(
            nn.Linear(self.in_feats, self.hidden_dim),
            nn.ReLU(),
        )
        if self.use_root_context:
            self.root_context = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
            )
        else:
            self.root_context = None

        self.depth_embedding = nn.Embedding(self.max_depth + 2, self.depth_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(self.hidden_dim + self.depth_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.encoder = nn.ModuleList(
            [
                DepthAwareTransformerBlock(
                    self.hidden_dim,
                    self.heads,
                    self.feedforward_dim,
                    self.dropout,
                    self.max_depth,
                    use_relative_depth_bias=self.use_relative_depth_bias,
                )
                for _ in range(self.layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        classifier_hidden = int(
            getattr(
                args,
                "depth_transformer_classifier_hidden_dim",
                self.hidden_dim,
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_hidden, self.num_classes),
        )

        class_weights = getattr(args, "classification_class_weights", None)
        if class_weights is None:
            self.register_buffer("classification_class_weights", torch.empty(0))
        else:
            self.register_buffer(
                "classification_class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )

        self._last_depth = None
        self._last_node_hidden = None
        self._last_graph_hidden = None
        self._last_attention = None

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def classification_loss(self, output, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.nll_loss(output, target.view(-1).long(), weight=weight)

    def physics_loss(self, U, S, D, true_state):
        return self.classifier[-1].weight.new_zeros(())

    def _root_indices(self, data):
        if hasattr(data, "ptr"):
            return data.ptr[:-1].to(device=data.x.device)
        batch = self._batch_vector(data)
        is_root = torch.ones(
            batch.size(0),
            dtype=torch.bool,
            device=batch.device,
        )
        is_root[1:] = batch[1:] != batch[:-1]
        return is_root.nonzero(as_tuple=False).view(-1)

    def _batch_vector(self, data):
        if hasattr(data, "batch"):
            return data.batch.long()
        return torch.zeros(
            data.x.size(0),
            dtype=torch.long,
            device=data.x.device,
        )

    def _num_graphs(self, data, batch):
        if hasattr(data, "num_graphs"):
            return int(data.num_graphs)
        if batch.numel() == 0:
            return 0
        return int(batch.max().item()) + 1

    def _directed_edge_index(self, data):
        return getattr(data, "directed_edge_index", data.edge_index).long()

    def _node_depths(self, data):
        num_nodes = data.x.size(0)
        depth = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=data.x.device,
        )
        roots = self._root_indices(data)
        depth[roots] = 0

        edge_index = self._directed_edge_index(data)
        if edge_index.numel() == 0:
            return depth

        src, dst = edge_index
        for _ in range(num_nodes):
            parent_depth = depth[src]
            candidate = parent_depth + 1
            update = (parent_depth >= 0) & (
                (depth[dst] < 0) | (candidate < depth[dst])
            )
            if not update.any():
                break
            depth[dst[update]] = candidate[update]
        return depth

    def _depth_indices(self, depth):
        return depth.long().clamp(-1, self.max_depth) + 1

    def _add_root_context(self, node_hidden, data):
        if self.root_context is None:
            return node_hidden
        roots = self._root_indices(data)
        batch = self._batch_vector(data)
        root_for_node = roots[batch]
        return self.root_context(
            torch.cat((node_hidden, node_hidden[root_for_node]), dim=-1)
        )

    def encode_nodes(self, data):
        batch = self._batch_vector(data)
        depth = self._node_depths(data)
        node_hidden = self.node_projection(data.x.float())
        node_hidden = self._add_root_context(node_hidden, data)
        depth_hidden = self.depth_embedding(self._depth_indices(depth))
        node_hidden = self.input_projection(
            torch.cat((node_hidden, depth_hidden), dim=-1)
        )

        dense_hidden, valid_mask = to_dense_batch(node_hidden, batch)
        dense_depth, _ = to_dense_batch(depth, batch, fill_value=-1)
        last_attention = None
        for block in self.encoder:
            dense_hidden, attention = block(
                dense_hidden,
                valid_mask,
                dense_depth,
            )
            last_attention = attention
        dense_hidden = self.output_norm(dense_hidden)
        encoded_nodes = dense_hidden[valid_mask]

        self._last_depth = depth.detach()
        self._last_node_hidden = encoded_nodes.detach()
        self._last_attention = (
            None if not self.log_attention else last_attention.detach()
        )
        return encoded_nodes, depth

    def encode_graph(self, data):
        batch = self._batch_vector(data)
        encoded_nodes, _ = self.encode_nodes(data)
        if self.pool == "root":
            graph_hidden = encoded_nodes[self._root_indices(data)]
        elif self.pool == "sum":
            graph_hidden = global_add_pool(encoded_nodes, batch)
        else:
            graph_hidden = global_mean_pool(encoded_nodes, batch)
        self._last_graph_hidden = graph_hidden.detach()
        return graph_hidden

    def forward(self, data):
        batch = self._batch_vector(data)
        graph_hidden = self.encode_graph(data)
        output_log_prob = F.log_softmax(
            self.classifier(graph_hidden),
            dim=-1,
        )
        batch_size = self._num_graphs(data, batch)
        placeholder = graph_hidden.new_zeros(batch_size, self.max_hop, 1)
        return output_log_prob, placeholder, placeholder, placeholder

    def __repr__(self):
        return self.__class__.__name__
