"""Edge-Bayesian GCN (EBGCN) adapted to the EIN graph-data interface."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from model.EIN_ResGCN import GCNConv as ResGCNConv


def _apply_batch_norm(batch_norm, x):
    """Use running statistics for a one-item 2-D training batch."""
    if batch_norm.training and x.dim() == 2 and x.size(0) == 1:
        return F.batch_norm(
            x,
            batch_norm.running_mean,
            batch_norm.running_var,
            batch_norm.weight,
            batch_norm.bias,
            training=False,
            momentum=0.0,
            eps=batch_norm.eps,
        )
    return batch_norm(x)


def _root_indices(data):
    if hasattr(data, 'ptr'):
        return data.ptr[:-1].to(device=data.batch.device)
    is_root = torch.ones(data.batch.size(0), dtype=torch.bool, device=data.batch.device)
    is_root[1:] = data.batch[1:] != data.batch[:-1]
    return is_root.nonzero(as_tuple=False).view(-1)


def _root_features(node_features, data):
    return node_features[_root_indices(data)][data.batch.long()]


def _top_down_edges(data):
    # ``directed_edge_index`` is retained by EIN even when the configured
    # message-passing graph is converted to undirected.
    return getattr(data, 'directed_edge_index', data.edge_index)


def _bottom_up_edges(data):
    edge_index = _top_down_edges(data)
    return edge_index.flip(0)


class _EdgeInference(nn.Module):
    """Author EBGCN's edge Bayesian inference network.

    The original code indexes ``x[row - 1]`` although PyG edge indices are
    zero-based.  This adaptation uses the actual source/destination endpoints.
    A reparameterized positive standard deviation is used in place of passing
    a log-variance directly to ``torch.normal``.
    """

    def __init__(self, hidden_features, edge_num):
        super().__init__()
        self.hidden_features = hidden_features
        self.edge_num = edge_num
        self.sim_network = self._create_network('sim_val')
        self.W_mean = self._create_network('W_mean')
        self.W_bias = self._create_network('W_bias')
        self.B_mean = self._create_network('B_mean')
        self.B_bias = self._create_network('B_bias')
        self.fc1 = nn.Linear(hidden_features, edge_num, bias=False)
        self.fc2 = nn.Linear(hidden_features, edge_num, bias=False)
        self.eval_loss = nn.KLDivLoss(reduction='batchmean')

    def _create_network(self, name):
        layers = OrderedDict()
        layers[name + 'conv0'] = nn.Conv1d(
            self.hidden_features, self.hidden_features, kernel_size=1, bias=False
        )
        layers[name + 'norm0'] = nn.BatchNorm1d(self.hidden_features)
        layers[name + 'relu0'] = nn.LeakyReLU()
        layers[name + 'conv_out'] = nn.Conv1d(
            self.hidden_features, 1, kernel_size=1
        )
        return nn.Sequential(layers)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return x.sum() * 0.0, x.new_zeros((0,))

        row, col = edge_index
        x_i = x[row].unsqueeze(2)
        x_j = x[col].unsqueeze(1)
        x_ij = torch.abs(x_i - x_j)

        sim_val = self.sim_network(x_ij)
        edge_pred = torch.sigmoid(self.fc1(sim_val))

        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = torch.log(
            sim_val.square() * torch.exp(w_bias.clamp(-20, 20))
            + torch.exp(b_bias.clamp(-20, 20))
            + 1e-8
        )
        std = F.softplus(0.5 * logit_var) + 1e-6
        edge_y = (
            logit_mean + std * torch.randn_like(logit_mean)
            if self.training
            else logit_mean
        )
        edge_y = torch.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)

        edge_loss = self.eval_loss(
            F.log_softmax(edge_pred, dim=-1), F.softmax(edge_y, dim=-1)
        )
        edge_weight = edge_pred.mean(dim=-1).squeeze(1)
        return edge_loss, edge_weight


class _RumorGCN(nn.Module):
    def __init__(self, in_feats, hidden_features, output_features, edge_num, infer_edges):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_features)
        self.conv2 = GCNConv(in_feats + hidden_features, output_features)
        self.bn1 = nn.BatchNorm1d(in_feats + hidden_features)
        self.infer_edges = infer_edges
        self.edge_infer = _EdgeInference(hidden_features, edge_num)

    def forward(self, data, edge_index):
        x_input = data.x.float()
        x = self.conv1(x_input, edge_index)
        x_first_layer = x

        if self.infer_edges:
            edge_loss, edge_weight = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_weight = None, None

        x = torch.cat((x, _root_features(x_input, data)), dim=1)
        x = F.relu(_apply_batch_norm(self.bn1, x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = torch.cat((x, _root_features(x_first_layer, data)), dim=1)
        return global_mean_pool(x, data.batch), edge_loss


class EBGCN(nn.Module):
    """Two-direction EBGCN classifier with TD and BU edge-inference losses."""

    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_features = int(getattr(args, 'ebgcn_hidden_dim', hidden_dim))
        self.output_features = int(
            getattr(args, 'ebgcn_output_dim', self.hidden_features)
        )
        self.edge_num = int(getattr(args, 'ebgcn_edge_num', 2))
        if self.edge_num < 2:
            raise ValueError('ebgcn_edge_num must be at least 2.')

        self.TDrumorGCN = _RumorGCN(
            in_feats,
            self.hidden_features,
            self.output_features,
            self.edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_td', True)),
        )
        self.BUrumorGCN = _RumorGCN(
            in_feats,
            self.hidden_features,
            self.output_features,
            self.edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_bu', True)),
        )
        self.fc = nn.Linear(
            (self.hidden_features + self.output_features) * 2, num_classes
        )

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, data):
        td_x, td_edge_loss = self.TDrumorGCN(data, _top_down_edges(data))
        bu_x, bu_edge_loss = self.BUrumorGCN(data, _bottom_up_edges(data))
        out = F.log_softmax(self.fc(torch.cat((bu_x, td_x), dim=1)), dim=1)
        return out, td_edge_loss, bu_edge_loss


class _RumorResGCN(nn.Module):
    """One directional edge-Bayesian ResGCN encoder."""

    def __init__(self, in_feats, hidden_features, edge_num, infer_edges, args):
        super().__init__()
        self.infer_edges = infer_edges
        self.residual = bool(getattr(args, 'skip_connection', True))
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.bn_feat = nn.BatchNorm1d(in_feats)
        self.conv_feat = ResGCNConv(in_feats, hidden_features, gfn=True)
        self.edge_infer = _EdgeInference(hidden_features, edge_num)

        num_conv_layers = int(
            getattr(
                args,
                'ebgcn_resgcn_num_conv_layers',
                getattr(args, 'n_layers_conv', 3),
            )
        )
        if num_conv_layers < 1:
            raise ValueError('EBGCN-ResGCN requires at least one graph convolution.')
        edge_norm = bool(getattr(args, 'edge_norm', True))
        self.bns_conv = nn.ModuleList(
            [nn.BatchNorm1d(hidden_features) for _ in range(num_conv_layers)]
        )
        self.convs = nn.ModuleList(
            [
                ResGCNConv(hidden_features, hidden_features, edge_norm=edge_norm)
                for _ in range(num_conv_layers)
            ]
        )

        num_fc_layers = int(getattr(args, 'n_layers_fc', 2))
        self.bns_fc = nn.ModuleList(
            [nn.BatchNorm1d(hidden_features) for _ in range(max(0, num_fc_layers - 1))]
        )
        self.lins = nn.ModuleList(
            [
                nn.Linear(hidden_features, hidden_features)
                for _ in range(max(0, num_fc_layers - 1))
            ]
        )
        self.bn_hidden = nn.BatchNorm1d(hidden_features)

        global_pool = str(getattr(args, 'global_pool', 'sum'))
        if 'sum' in global_pool:
            self.global_pool = global_add_pool
        elif 'mean' in global_pool:
            self.global_pool = global_mean_pool
        else:
            raise ValueError('global_pool must contain "sum" or "mean".')
        self.gating = (
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1),
                nn.Sigmoid(),
            )
            if 'gating' in global_pool
            else None
        )

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)

    def forward(self, data, edge_index):
        x = F.relu(
            self.conv_feat(
                _apply_batch_norm(self.bn_feat, data.x.float()), edge_index
            )
        )
        if self.infer_edges:
            edge_loss, edge_weight = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_weight = None, None

        for batch_norm, conv in zip(self.bns_conv, self.convs):
            residual = F.relu(
                conv(
                    _apply_batch_norm(batch_norm, x),
                    edge_index,
                    edge_weight=edge_weight,
                )
            )
            x = x + residual if self.residual else residual

        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, data.batch)
        for batch_norm, linear in zip(self.bns_fc, self.lins):
            x = F.relu(linear(_apply_batch_norm(batch_norm, x)))
        x = _apply_batch_norm(self.bn_hidden, x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x, edge_loss


class EBGCNResGCN(nn.Module):
    """EBGCN whose independent TD/BU encoders use a ResGCN backbone."""

    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_features = int(getattr(args, 'ebgcn_hidden_dim', hidden_dim))
        self.edge_num = int(getattr(args, 'ebgcn_edge_num', 2))
        if self.edge_num < 2:
            raise ValueError('ebgcn_edge_num must be at least 2.')

        self.TDrumorGCN = _RumorResGCN(
            in_feats,
            self.hidden_features,
            self.edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_td', True)),
            args,
        )
        self.BUrumorGCN = _RumorResGCN(
            in_feats,
            self.hidden_features,
            self.edge_num,
            bool(getattr(args, 'ebgcn_edge_infer_bu', True)),
            args,
        )
        self.fc = nn.Linear(self.hidden_features * 2, num_classes)

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def forward(self, data):
        td_x, td_edge_loss = self.TDrumorGCN(data, _top_down_edges(data))
        bu_x, bu_edge_loss = self.BUrumorGCN(data, _bottom_up_edges(data))
        out = F.log_softmax(self.fc(torch.cat((bu_x, td_x), dim=1)), dim=1)
        return out, td_edge_loss, bu_edge_loss
