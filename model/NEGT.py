"""NEGT model adapted from JYZHU03/NEGT for the EIN training interface.

The architecture and attention-mask construction intentionally follow the
upstream implementation.  Only dataset/training globals are replaced by
constructor arguments and model state so that it can run in this repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class NEGTTransformerBlock(nn.Module):
    """Upstream ``Atten_transformer`` with its original active operations."""

    def __init__(self, out_feats):
        super().__init__()
        self.heads = 4
        self.attn_dropout = 0.0
        self.dropout = 0.0
        self.log_attn_weights = False
        self.layer_norm = False
        self.batch_norm = True
        self.is_need_attn = False

        # ``norm1_local`` is unused upstream, but remains part of the module
        # so parameterization matches the released Atten_transformer class.
        self.norm1_local = nn.BatchNorm1d(out_feats)
        self.norm1_attn = nn.BatchNorm1d(out_feats)
        self.dropout_local = nn.Dropout(self.dropout)
        self.dropout_attn = nn.Dropout(self.dropout)
        self.ff_linear1 = nn.Linear(out_feats, out_feats)
        self.ff_linear2 = nn.Linear(out_feats, out_feats)
        self.act_fn_ff = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(out_feats)
        self.ff_dropout1 = nn.Dropout(self.dropout)
        self.ff_dropout2 = nn.Dropout(self.dropout)
        self.self_attn = nn.MultiheadAttention(
            out_feats,
            self.heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

    def forward(self, x, batchindex, attn_mask=None):
        h_in1 = x
        h_out_list = [x]
        h_dense, key_padding_mask = to_dense_batch(x, batchindex)
        key_padding_mask = ~key_padding_mask

        x = self.self_attn(
            h_dense,
            h_dense,
            h_dense,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        h_attn = x[~key_padding_mask]
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn
        h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        h = sum(h_out_list)
        h = h + self._ff_block(h)
        return self.norm2(h)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


def _source_root_decay(adjacency, beta=0.5):
    """Torch equivalent of upstream ``scipy.dijkstra`` decay.

    ``add_relation_edges`` first symmetrizes adjacency, so breadth-first search
    returns the same unit-weight shortest-path distances while keeping the
    calculation on the active device.
    """

    num_nodes = adjacency.size(0)
    distance = torch.full(
        (num_nodes,), float('inf'), dtype=adjacency.dtype, device=adjacency.device
    )
    if num_nodes == 0:
        return distance

    distance[0] = 0
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=adjacency.device)
    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=adjacency.device)
    frontier[0] = True
    depth = 0
    adjacency_bool = adjacency > 0

    while frontier.any():
        visited |= frontier
        next_frontier = adjacency_bool[frontier].any(dim=0) & ~visited
        depth += 1
        distance[next_frontier] = depth
        frontier = next_frontier

    return torch.exp(-beta * distance)


def _source_relation_attention(data):
    """Reproduce ``utils.mask_graph.add_relation_edges`` from NEGT.

    In particular, the second-order relation retains its diagonal entries, as
    in the released code.  This is intentionally different from the previous
    repository implementation, which explicitly removed that diagonal.
    """

    edge_index, batch = data.edge_index, data.batch
    batch_size = int(batch[-1].item()) + 1
    graph_ptrs = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=batch.device),
            batch.bincount().cumsum(dim=0),
        ]
    )
    max_nodes = int((graph_ptrs[1:] - graph_ptrs[:-1]).max().item())
    adjacency_matrices = []

    for graph_id in range(batch_size):
        start_ptr = graph_ptrs[graph_id]
        graph_edges = edge_index[
            (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
        ]
        adjacency = torch.zeros(
            (max_nodes, max_nodes), dtype=torch.float32, device=edge_index.device
        )
        if graph_edges.numel() > 0:
            source = graph_edges[0] - start_ptr
            target = graph_edges[1] - start_ptr
            adjacency[source, target] = 1
            adjacency[target, source] = 1

        adjacency_square = torch.mm(adjacency, adjacency)
        second_order = ((adjacency_square > 0) & (adjacency == 0)).float() * 2.0
        decay = _source_root_decay(adjacency, beta=0.5)
        relation = adjacency + second_order
        relation_bias = torch.zeros_like(relation)
        non_zero = relation != 0
        relation_bias[non_zero] = torch.exp(1.0 / relation[non_zero])
        adjacency_matrices.append(
            relation_bias * decay.view(-1, 1) * decay.view(1, -1)
        )

    return torch.stack(adjacency_matrices)


def build_negt_attention_bias(data, node_attention, scale=1000.1):
    """Reproduce the four-head mask order in upstream ``get_attention_mask``."""

    full_attention = NEGT.get_full_attention(data.batch, data.edge_index.device)
    edge_attention = NEGT.lift_node_att_to_edge_att(
        node_attention, full_attention, data.batch
    )
    relation_attention = _source_relation_attention(data)
    unrestricted = torch.zeros_like(relation_attention)

    # The upstream code concatenates heads rather than interleaving them by
    # graph.  Preserve that order for numerical alignment with its checkpoints.
    return torch.cat(
        [edge_attention, relation_attention, unrestricted, unrestricted], dim=0
    ) * scale


class NEGT(torch.nn.Module):
    """NEGT with the upstream model equations and EIN-compatible lifecycle."""

    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.num_features = in_feats
        self.num_classes = num_classes
        self.nhid = int(getattr(args, 'negt_hidden_dim', hidden_dim))
        self.concat = bool(getattr(args, 'negt_concat', False))
        self.usegnn = bool(getattr(args, 'negt_use_gnn', False))
        self.dropout = float(getattr(args, 'negt_dropout', 0.5))
        configured_heads = int(getattr(args, 'negt_heads', 4))
        if configured_heads != 4:
            raise ValueError('The released NEGT implementation requires negt_heads=4.')
        self.attention_scale = float(getattr(args, 'negt_attention_scale', 1000.1))
        if self.attention_scale != 1000.1:
            raise ValueError(
                'The released NEGT implementation requires negt_attention_scale=1000.1.'
            )
        self.current_epoch = 0

        self.conv1 = GATConv(self.num_features, self.nhid * 2)
        self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)
        self.liner1 = Linear(self.num_features, self.nhid * 2)
        self.liner2 = Linear(self.nhid * 2, self.nhid * 2)
        self.liner3 = Linear(self.nhid * 2, 1)
        self.Atten_transformer1 = NEGTTransformerBlock(self.nhid * 2)
        self.Atten_transformer2 = NEGTTransformerBlock(self.nhid * 2)

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

    @property
    def linear1(self):
        return self.liner1

    @property
    def linear2(self):
        return self.liner2

    @property
    def linear3(self):
        return self.liner3

    @property
    def transformer1(self):
        return self.Atten_transformer1

    @property
    def transformer2(self):
        return self.Atten_transformer2

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        if self.usegnn:
            x_att = F.selu(self.conv1(x, edge_index))
            x_att = F.selu(self.conv2(x_att, edge_index))
            att_log_logits = torch.sigmoid(F.selu(self.liner3(x_att)))
            att = self.sampling(att_log_logits, self.training)
            edge_att = self.lift_node_att_to_edge_att_gnn(att, edge_index)
            x = F.selu(self.conv1(x, edge_index, edge_att))
            x = F.selu(self.conv2(x, edge_index, edge_att))
        else:
            x_att = F.selu(self.liner1(x))
            x_att = self.Atten_transformer1(x_att, batch, None)
            x_att = F.selu(self.liner2(x_att))
            x_att = self.Atten_transformer2(x_att, batch, None)

            att_log_logits = torch.sigmoid(F.selu(self.liner3(x_att)))
            att = self.sampling(att_log_logits, self.training)
            attn_mask = build_negt_attention_bias(data, att, self.attention_scale)

            x = F.selu(self.liner1(x))
            x = self.Atten_transformer1(x, batch, attn_mask)
            x = F.selu(self.liner2(x))
            x = self.Atten_transformer2(x, batch, attn_mask)

        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.concat:
            root_index = data.ptr[:-1] if hasattr(data, 'ptr') else self._root_index(batch)
            news = F.relu(self.fc0(data.x[root_index].float()))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        out = F.log_softmax(self.fc2(x), dim=-1)
        r = self.fix_r if self.fix_r else self.get_r()
        info_loss = (
            att * torch.log(att / r + 1e-6)
            + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)
        ).mean()
        return out, info_loss

    def sampling(self, att_log_logits, training):
        return self.concrete_sample(att_log_logits, temp=1, training=training)

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index, batch):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        # Avoid upstream ``squeeze``: its value is the same for normal batches
        # but it collapses dimensions for a one-graph batch.
        return to_dense_adj(edge_index, batch, edge_attr=edge_att).squeeze(-1)

    @staticmethod
    def lift_node_att_to_edge_att_gnn(node_att, edge_index):
        return node_att[edge_index[0]] * node_att[edge_index[1]]

    @staticmethod
    def get_full_attention(batch, device):
        edge_index_full = torch.empty((2, 0), dtype=torch.long, device=device)
        for graph_id in batch.unique():
            nodes = (batch == graph_id).nonzero(as_tuple=True)[0]
            edge_index = torch.combinations(nodes, r=2).to(device)
            edge_index_full = torch.cat(
                [edge_index_full, edge_index.t(), edge_index.flip([1]).t()], dim=1
            )
            edge_index_full = torch.cat(
                [edge_index_full, torch.stack([nodes, nodes])], dim=1
            )
        return edge_index_full

    @staticmethod
    def _root_index(batch):
        is_root = torch.ones(batch.size(0), dtype=torch.bool, device=batch.device)
        is_root[1:] = batch[1:] != batch[:-1]
        return is_root.nonzero(as_tuple=False).view(-1)

    def get_r(self):
        r = self.init_r - self.current_epoch // self.decay_interval * self.decay_r
        return max(r, self.final_r)
