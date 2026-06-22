import copy
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, Parameter
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add, scatter_mean


class GCNConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        bias=True,
        edge_norm=True,
        gfn=False,
    ):
        super(GCNConv, self).__init__('add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        loop_weight = torch.full(
            (num_nodes,),
            1 if not improved else 2,
            dtype=edge_weight.dtype,
            device=edge_weight.device,
        )
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, x.size(0), edge_weight, self.improved, x.dtype
                )
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class RAGCLResGCN(torch.nn.Module):
    """RAGCL supervised ResGCN baseline without EIN epidemiology branch."""

    def __init__(
        self,
        dataset=None,
        num_classes=2,
        hidden=128,
        num_feat_layers=1,
        num_conv_layers=3,
        num_fc_layers=2,
        gfn=False,
        collapse=False,
        residual=False,
        res_branch='BNConvReLU',
        global_pool='sum',
        dropout=0,
        edge_norm=True,
    ):
        super(RAGCLResGCN, self).__init__()
        assert num_feat_layers == 1, 'more feat layers are not now supported'
        self.num_classes = num_classes
        self.conv_residual = residual
        self.fc_residual = False
        self.res_branch = res_branch
        self.collapse = collapse
        self.dropout = dropout

        assert 'sum' in global_pool or 'mean' in global_pool, global_pool
        self.global_pool = global_add_pool if 'sum' in global_pool else global_mean_pool
        gconv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = dataset.num_features
        self.use_xg = False
        if len(dataset) > 0 and hasattr(dataset[0], 'xg'):
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)

        if collapse:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            self.gating = self._build_gating(hidden_in, global_pool)
            for _ in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, self.num_classes)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)
            self.gating = self._build_gating(hidden, global_pool)
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()

            if self.res_branch == 'resnet':
                for _ in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=True))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(gconv(hidden, hidden))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=True))
            else:
                for _ in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(gconv(hidden, hidden))

            self.bn_hidden = BatchNorm1d(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for _ in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, self.num_classes)

        self.proj_head = nn.Sequential(
            Linear(hidden, hidden), nn.ReLU(inplace=True), Linear(hidden, hidden)
        )

        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0.0001)

    @staticmethod
    def _build_gating(hidden, global_pool):
        if 'gating' not in global_pool:
            return None
        return torch.nn.Sequential(
            Linear(hidden, hidden),
            torch.nn.ReLU(),
            Linear(hidden, 1),
            torch.nn.Sigmoid(),
        )

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _xg(self, data):
        if not self.use_xg:
            return None
        xg = self.bn1_xg(data.xg)
        xg = F.relu(self.lin1_xg(xg))
        xg = self.bn2_xg(xg)
        return F.relu(self.lin2_xg(xg))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xg = self._xg(data)
        if self.collapse:
            return self.forward_collapse(x, edge_index, batch, xg)
        if self.res_branch == 'BNConvReLU':
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        if self.res_branch == 'BNReLUConv':
            return self.forward_BNReLUConv(x, edge_index, batch, xg)
        if self.res_branch == 'ConvReLUBN':
            return self.forward_ConvReLUBN(x, edge_index, batch, xg)
        if self.res_branch == 'resnet':
            return self.forward_resnet(x, edge_index, batch, xg)
        raise ValueError('Unknown res_branch {}'.format(self.res_branch))

    def forward_collapse(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_BNReLUConv(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(self.bns_conv[i](x))
            x_ = conv(x_, edge_index)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_ if self.fc_residual else x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_ConvReLUBN(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        x = self.bn_hidden(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(conv(x, edge_index))
            x_ = self.bns_conv[i](x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(lin(x))
            x_ = self.bns_fc[i](x_)
            x = x + x_ if self.fc_residual else x_
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_resnet(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i in range(len(self.convs) // 3):
            x_ = x
            x_ = F.relu(self.bns_conv[i * 3](x_))
            x_ = self.convs[i * 3](x_, edge_index)
            x_ = F.relu(self.bns_conv[i * 3 + 1](x_))
            x_ = self.convs[i * 3 + 1](x_, edge_index)
            x_ = F.relu(self.bns_conv[i * 3 + 2](x_))
            x_ = self.convs[i * 3 + 2](x_, edge_index)
            x = x + x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_graphcl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xg = self._xg(data)
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.proj_head(x)

    def loss_graphcl(self, x1, x2, mean=True):
        return _graphcl_loss(x1, x2, mean)


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, tddroprate=0.0):
        super(TDrumorGCN, self).__init__()
        self.tddroprate = tddroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, data.edge_index
        edge_index = self._drop_edges(edge_index, self.tddroprate, device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = torch.eq(data.batch, num_batch)
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = torch.eq(data.batch, num_batch)
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        return scatter_mean(x, data.batch, dim=0)

    @staticmethod
    def _drop_edges(edge_index, droprate, device):
        if droprate <= 0:
            return edge_index
        edge_index_list = edge_index.tolist()
        length = len(edge_index_list[0])
        keep_count = int(length * (1 - droprate))
        if keep_count <= 0:
            return edge_index
        poslist = sorted(random.sample(range(length), keep_count))
        row = list(np.array(edge_index_list[0])[poslist])
        col = list(np.array(edge_index_list[1])[poslist])
        return torch.LongTensor([row, col]).to(device)


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, budroprate=0.0):
        super(BUrumorGCN, self).__init__()
        self.budroprate = budroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]
        edge_index = TDrumorGCN._drop_edges(edge_index, self.budroprate, device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = torch.eq(data.batch, num_batch)
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = torch.eq(data.batch, num_batch)
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        return scatter_mean(x, data.batch, dim=0)


class RAGCLBiGCN(torch.nn.Module):
    """RAGCL supervised BiGCN baseline without EIN epidemiology branch."""

    def __init__(
        self,
        in_feats,
        hid_feats,
        out_feats,
        num_classes,
        tddroprate=0.0,
        budroprate=0.0,
    ):
        super(RAGCLBiGCN, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, tddroprate)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, budroprate)
        self.proj_head = torch.nn.Linear((out_feats + hid_feats) * 2, out_feats)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_classes)

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, data):
        td_x = self.TDrumorGCN(data)
        bu_x = self.BUrumorGCN(data)
        x = torch.cat((bu_x, td_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def forward_graphcl(self, data):
        td_x = self.TDrumorGCN(data)
        bu_x = self.BUrumorGCN(data)
        x = torch.cat((bu_x, td_x), 1)
        return self.proj_head(x)

    def loss_graphcl(self, x1, x2, mean=True):
        return _graphcl_loss(x1, x2, mean)


def _graphcl_loss(x1, x2, mean=True):
    temperature = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2)
    sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss)
    if mean:
        loss = loss.mean()
    return loss
