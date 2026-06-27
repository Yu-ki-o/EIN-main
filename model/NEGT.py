import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class NEGTTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super(NEGTTransformerBlock, self).__init__()
        self.heads = heads
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1_attn = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_attn = nn.Dropout(0.0)
        self.ff_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.ff_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ff_dropout1 = nn.Dropout(0.0)
        self.ff_dropout2 = nn.Dropout(0.0)

    def forward(self, x, batch, attn_mask=None):
        h_in = x
        h_dense, valid_mask = to_dense_batch(x, batch)
        key_padding_mask = torch.zeros(
            valid_mask.size(),
            dtype=h_dense.dtype,
            device=h_dense.device,
        )
        key_padding_mask = key_padding_mask.masked_fill(~valid_mask, float('-inf'))

        h_attn = self.self_attn(
            h_dense,
            h_dense,
            h_dense,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        h_attn = h_attn[valid_mask]
        h_attn = self.dropout_attn(h_attn)
        h_attn = self.norm1_attn(h_in + h_attn)

        h = h_in + h_attn
        h = h + self._ff_block(h)
        h = self.norm2(h)
        return h

    def _ff_block(self, x):
        x = self.ff_dropout1(F.relu(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


def _dense_node_attention(node_attention, batch):
    dense_attention, valid_mask = to_dense_batch(node_attention, batch)
    edge_attention = dense_attention * dense_attention.transpose(1, 2)
    edge_attention = edge_attention * valid_mask.unsqueeze(1).float()
    edge_attention = edge_attention * valid_mask.unsqueeze(2).float()
    return edge_attention


def _root_decay(adj, num_nodes, beta=0.5):
    device = adj.device
    dist = torch.full((adj.size(0),), float('inf'), device=device)
    if num_nodes == 0:
        return torch.zeros_like(dist)

    dist[0] = 0
    visited = torch.zeros((adj.size(0),), dtype=torch.bool, device=device)
    frontier = torch.tensor([0], dtype=torch.long, device=device)
    depth = 0
    adj_bool = adj[:num_nodes, :num_nodes] > 0

    while frontier.numel() > 0:
        visited[frontier] = True
        neighbors = adj_bool[frontier].any(dim=0)
        next_frontier = (neighbors & ~visited[:num_nodes]).nonzero(as_tuple=False).view(-1)
        depth += 1
        if next_frontier.numel() > 0:
            dist[next_frontier] = depth
        frontier = next_frontier

    decay = torch.exp(-beta * dist)
    decay[~torch.isfinite(decay)] = 0
    return decay


def _relation_attention_bias(data, max_nodes):
    edge_index = data.edge_index
    batch = data.batch
    device = edge_index.device
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    node_counts = torch.bincount(batch, minlength=batch_size)
    relation = torch.zeros(
        (batch_size, max_nodes, max_nodes),
        dtype=torch.float32,
        device=device,
    )

    for graph_id in range(batch_size):
        num_nodes = int(node_counts[graph_id].item())
        if num_nodes == 0:
            continue

        graph_nodes = (batch == graph_id).nonzero(as_tuple=False).view(-1)
        start = int(graph_nodes[0].item())
        graph_edge_mask = batch[edge_index[0]] == graph_id
        graph_edges = edge_index[:, graph_edge_mask] - start
        valid_edges = (
            (graph_edges[0] >= 0)
            & (graph_edges[0] < num_nodes)
            & (graph_edges[1] >= 0)
            & (graph_edges[1] < num_nodes)
        )
        graph_edges = graph_edges[:, valid_edges]

        adj = torch.zeros((max_nodes, max_nodes), dtype=torch.float32, device=device)
        if graph_edges.numel() > 0:
            adj[graph_edges[0], graph_edges[1]] = 1.0
            adj[graph_edges[1], graph_edges[0]] = 1.0

        second_order = torch.mm(adj, adj)
        second_order = ((second_order > 0) & (adj == 0)).float() * 2.0
        second_order.fill_diagonal_(0.0)

        relation_weight = adj + second_order
        positive = relation_weight > 0
        proximity = torch.zeros_like(relation_weight)
        proximity[positive] = torch.exp(1.0 / relation_weight[positive])

        decay = _root_decay(adj, num_nodes, beta=0.5)
        relation[graph_id] = proximity * decay.view(-1, 1) * decay.view(1, -1)

    return relation


def build_negt_attention_bias(data, node_attention, heads=4, scale=1000.1):
    edge_attention = _dense_node_attention(node_attention, data.batch)
    batch_size, max_nodes, _ = edge_attention.size()
    relation_attention = _relation_attention_bias(data, max_nodes)
    unrestricted = torch.zeros_like(edge_attention)

    head_biases = [edge_attention, relation_attention]
    while len(head_biases) < heads:
        head_biases.append(unrestricted)
    head_biases = head_biases[:heads]

    attention_bias = torch.stack(head_biases, dim=1) * scale
    return attention_bias.reshape(batch_size * heads, max_nodes, max_nodes)


class NEGT(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super(NEGT, self).__init__()
        self.args = args
        self.device = device
        self.num_features = in_feats
        self.num_classes = num_classes
        self.nhid = int(getattr(args, 'negt_hidden_dim', hidden_dim))
        self.concat = bool(getattr(args, 'negt_concat', False))
        self.use_gnn = bool(getattr(args, 'negt_use_gnn', False))
        self.dropout = float(getattr(args, 'negt_dropout', getattr(args, 'dropout', 0.5)))
        self.heads = int(getattr(args, 'negt_heads', 4))
        self.attention_scale = float(getattr(args, 'negt_attention_scale', 1000.1))
        self.current_epoch = 0

        self.conv1 = GATConv(self.num_features, self.nhid * 2)
        self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

        self.linear1 = Linear(self.num_features, self.nhid * 2)
        self.linear2 = Linear(self.nhid * 2, self.nhid * 2)
        self.linear3 = Linear(self.nhid * 2, 1)

        self.transformer1 = NEGTTransformerBlock(self.nhid * 2, heads=self.heads)
        self.transformer2 = NEGTTransformerBlock(self.nhid * 2, heads=self.heads)

        self.fc1 = Linear(self.nhid * 2, self.nhid)
        if self.concat:
            self.fc0 = Linear(self.num_features, self.nhid)
            self.fc1 = Linear(self.nhid * 2, self.nhid)
        self.fc2 = Linear(self.nhid, self.num_classes)

        self.fix_r = getattr(args, 'negt_fix_r', False)
        self.decay_interval = int(getattr(args, 'negt_decay_interval', 10))
        self.decay_r = float(getattr(args, 'negt_decay_r', 0.1))
        self.final_r = float(getattr(args, 'negt_final_r', 0.8))
        self.init_r = float(getattr(args, 'negt_init_r', 0.9))

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        training = self.training

        if self.use_gnn:
            x_att = F.selu(self.conv1(x, edge_index))
            x_att = F.selu(self.conv2(x_att, edge_index))
            attention_logits = torch.sigmoid(F.selu(self.linear3(x_att)))
            node_attention = self.sampling(attention_logits, training)
            edge_attention = self.lift_node_att_to_edge_att_gnn(node_attention, edge_index)
            x = F.selu(self.conv1(x, edge_index, edge_attention))
            x = F.selu(self.conv2(x, edge_index, edge_attention))
        else:
            x_att = F.selu(self.linear1(x))
            x_att = self.transformer1(x_att, batch, None)
            x_att = F.selu(self.linear2(x_att))
            x_att = self.transformer2(x_att, batch, None)

            attention_logits = torch.sigmoid(F.selu(self.linear3(x_att)))
            node_attention = self.sampling(attention_logits, training)
            attention_bias = build_negt_attention_bias(
                data,
                node_attention,
                heads=self.heads,
                scale=self.attention_scale,
            )

            x = F.selu(self.linear1(x))
            x = self.transformer1(x, batch, attention_bias)
            x = F.selu(self.linear2(x))
            x = self.transformer2(x, batch, attention_bias)

        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=training)

        if self.concat:
            root_index = data.ptr[:-1] if hasattr(data, 'ptr') else self._root_index(batch)
            news = data.x[root_index].float()
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        out = F.log_softmax(self.fc2(x), dim=-1)
        return out, self.information_loss(node_attention)

    def sampling(self, attention_logits, training):
        return self.concrete_sample(attention_logits, temp=1.0, training=training)

    @staticmethod
    def concrete_sample(attention_logits, temp, training):
        if training:
            random_noise = torch.empty_like(attention_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            return ((attention_logits + random_noise) / temp).sigmoid()
        return attention_logits.sigmoid()

    @staticmethod
    def lift_node_att_to_edge_att_gnn(node_attention, edge_index):
        return node_attention[edge_index[0]] * node_attention[edge_index[1]]

    @staticmethod
    def _root_index(batch):
        is_root = torch.ones(batch.size(0), dtype=torch.bool, device=batch.device)
        is_root[1:] = batch[1:] != batch[:-1]
        return is_root.nonzero(as_tuple=False).view(-1)

    def get_r(self):
        if self.fix_r is not False:
            return float(self.fix_r)
        r = self.init_r - self.current_epoch // self.decay_interval * self.decay_r
        return max(r, self.final_r)

    def information_loss(self, node_attention):
        r = self.get_r()
        eps = 1e-6
        loss = (
            node_attention * torch.log(node_attention / r + eps)
            + (1 - node_attention)
            * torch.log((1 - node_attention) / (1 - r + eps) + eps)
        )
        return loss.mean()
