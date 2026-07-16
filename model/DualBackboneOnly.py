import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import Parameter
from torch_geometric.nn import (
    GCNConv as PyGGCNConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops


class _MatchedResGCNConv(MessagePassing):
    """Local copy of the custom ResGCN GCNConv used in EIN_ResGCN."""

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
        super().__init__("add")
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
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),),
                dtype=dtype,
                device=edge_index.device,
            )
        edge_weight = edge_weight.view(-1)
        if edge_weight.size(0) != edge_index.size(1):
            raise ValueError("edge_weight must match edge_index size")

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
        deg = edge_weight.new_zeros(num_nodes)
        deg.index_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = _MatchedResGCNConv.norm(
                    edge_index,
                    x.size(0),
                    edge_weight,
                    self.improved,
                    x.dtype,
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


class _BackboneOnlyBase(nn.Module):
    """
    Plain propagation-backbone baseline matched to the backbone code used by
    BiGCN_UncertaintySemanticChange and ResGCN_UncertaintySemanticChange.

    The model intentionally excludes stance routing, semantic parity, dual
    views, semantic change, DS, conflict, and semantic-tree branches.
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
        self.out_feats = int(out_feats)
        self.num_classes = int(num_classes)
        self.max_hop = int(getattr(args, "max_hop", 1))
        self.dropout = float(getattr(args, "dropout", 0.0))

        pool_name = str(getattr(args, "global_pool", "mean")).lower()
        self.pool_is_sum = "sum" in pool_name
        self.global_pool = (
            global_add_pool if self.pool_is_sum else global_mean_pool
        )

        self._build_backbone(in_feats, hid_feats, out_feats, args)
        self._build_classifier(args)
        self._last_graph_hidden = None

    def _build_backbone(self, in_feats, hid_feats, out_feats, args):
        raise NotImplementedError

    def _encode_nodes(self, data):
        raise NotImplementedError

    def _build_classifier(self, args):
        self.classification_head_mode = str(
            getattr(args, "classification_head_mode", "fusion")
        ).strip().lower()
        valid_head_modes = {"fusion", "branch_sum"}
        if self.classification_head_mode not in valid_head_modes:
            raise ValueError(
                "classification_head_mode must be one of {}, got {}".format(
                    sorted(valid_head_modes),
                    self.classification_head_mode,
                )
            )

        if self.classification_head_mode == "branch_sum":
            self.classifier = nn.Linear(
                self.hidden_dim,
                self.num_classes,
            )
            self.fusion = nn.Identity()
        else:
            fusion_hidden = int(
                getattr(
                    self.args,
                    "classification_fusion_hidden_dim",
                    self.hidden_dim * 2,
                )
            )
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(fusion_hidden, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim),
            )
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        class_weights = getattr(self.args, "classification_class_weights", None)
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

    def classification_loss(self, output, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.nll_loss(output, target.view(-1).long(), weight=weight)

    def physics_loss(self, U, S, D, true_state):
        return self.classifier.weight.new_zeros(())

    def auxiliary_loss(self):
        return self.classifier.weight.new_zeros(())

    def _root_indices(self, data):
        if hasattr(data, "ptr"):
            return data.ptr[:-1].to(device=data.batch.device)
        is_root = torch.ones(
            data.batch.size(0),
            dtype=torch.bool,
            device=data.batch.device,
        )
        if data.batch.numel() > 1:
            is_root[1:] = data.batch[1:] != data.batch[:-1]
        return is_root.nonzero(as_tuple=False).view(-1)

    def _extend_root_features(self, node_features, data):
        roots = self._root_indices(data)
        return node_features[roots][data.batch.long()]

    def _placeholder_sequence(self, data, graph_hidden):
        if hasattr(data, "num_graphs"):
            batch_size = int(data.num_graphs)
        elif data.batch.numel() == 0:
            batch_size = 0
        else:
            batch_size = int(data.batch.max().item()) + 1
        return graph_hidden.new_zeros(batch_size, self.max_hop, 1)

    def forward(self, data):
        node_hidden = self._encode_nodes(data)
        graph_hidden = self.global_pool(node_hidden, data.batch)
        self._last_graph_hidden = graph_hidden.detach()

        fused = self.fusion(graph_hidden)
        logits = self.classifier(fused)
        output = F.log_softmax(logits, dim=-1)
        placeholder = self._placeholder_sequence(data, graph_hidden)
        return output, placeholder, placeholder, placeholder


class BiGCN_BackboneOnly(_BackboneOnlyBase):
    """
    BiGCN backbone-only baseline matched to BiGCN_UncertaintySemanticChange.

    It uses the same TD/BU GCN shape, root feature extension, direction
    fusion, dropout, graph pooling, and n_layers_conv setting as the
    semantic-change model's BiGCN view encoder.
    """

    def _build_backbone(self, in_feats, hid_feats, out_feats, args):
        self.bigcn_num_layers = max(
            2,
            int(getattr(args, "n_layers_conv", 2)),
        )
        self.td_conv1 = PyGGCNConv(in_feats, hid_feats)
        self.td_conv2 = PyGGCNConv(hid_feats + in_feats, out_feats)
        self.bu_conv1 = PyGGCNConv(in_feats, hid_feats)
        self.bu_conv2 = PyGGCNConv(hid_feats + in_feats, out_feats)
        self.td_extra_convs = nn.ModuleList(
            [
                PyGGCNConv(out_feats + out_feats, out_feats)
                for _ in range(self.bigcn_num_layers - 2)
            ]
        )
        self.bu_extra_convs = nn.ModuleList(
            [
                PyGGCNConv(out_feats + out_feats, out_feats)
                for _ in range(self.bigcn_num_layers - 2)
            ]
        )
        branch_dim = out_feats + hid_feats
        self.direction_fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, hid_feats),
            nn.ReLU(),
        )

    def _reverse_edges(self, edge_index):
        return torch.stack((edge_index[1], edge_index[0]), dim=0)

    def _encode_bigcn_direction(
        self,
        data,
        edge_index,
        edge_weight,
        conv1,
        conv2,
        extra_convs=None,
    ):
        raw_nodes = data.x.float()
        hidden_first = conv1(
            raw_nodes,
            edge_index,
            edge_weight=edge_weight,
        )
        root_raw = self._extend_root_features(raw_nodes, data)
        hidden = torch.cat((hidden_first, root_raw), dim=-1)
        hidden = F.relu(hidden)
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        hidden = conv2(hidden, edge_index, edge_weight=edge_weight)
        hidden = F.relu(hidden)
        root_hidden = self._extend_root_features(hidden_first, data)
        for conv in extra_convs or []:
            root_current = self._extend_root_features(hidden, data)
            hidden = torch.cat((hidden, root_current), dim=-1)
            hidden = F.relu(hidden)
            hidden = F.dropout(
                hidden,
                p=self.dropout,
                training=self.training,
            )
            hidden = conv(hidden, edge_index, edge_weight=edge_weight)
            hidden = F.relu(hidden)
        return torch.cat((hidden, root_hidden), dim=-1)

    def _encode_nodes(self, data):
        edge_weight = data.x.new_ones(data.edge_index.size(1))
        top_down = self._encode_bigcn_direction(
            data,
            data.edge_index,
            edge_weight,
            self.td_conv1,
            self.td_conv2,
            self.td_extra_convs,
        )
        bottom_up = self._encode_bigcn_direction(
            data,
            self._reverse_edges(data.edge_index),
            edge_weight,
            self.bu_conv1,
            self.bu_conv2,
            self.bu_extra_convs,
        )
        return self.direction_fusion(torch.cat((top_down, bottom_up), dim=-1))

    def __repr__(self):
        return self.__class__.__name__


class ResGCN_BackboneOnly(_BackboneOnlyBase):
    """
    ResGCN backbone-only baseline matched to ResGCN_UncertaintySemanticChange.

    It uses the same initial GFN projection, residual GCN stack, BatchNorm,
    edge_norm flag, dropout, and graph pooling configuration as the
    semantic-change model's ResGCN view encoder.
    """

    def _build_backbone(self, in_feats, hid_feats, out_feats, args):
        self.bn_feat = BatchNorm1d(in_feats)
        self.conv_feat = _MatchedResGCNConv(in_feats, hid_feats, gfn=True)
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        edge_norm = bool(getattr(args, "edge_norm", True))
        for _ in range(max(1, int(getattr(args, "n_layers_conv", 3)))):
            self.bns_conv.append(BatchNorm1d(hid_feats))
            self.convs.append(
                _MatchedResGCNConv(
                    hid_feats,
                    hid_feats,
                    edge_norm=edge_norm,
                )
            )
        for module in self.modules():
            if isinstance(module, BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)

    def _encode_nodes(self, data):
        edge_weight = data.x.new_ones(data.edge_index.size(1))
        hidden = self.bn_feat(data.x.float())
        hidden = F.relu(
            self.conv_feat(
                hidden,
                data.edge_index,
                edge_weight=edge_weight,
            )
        )
        for batch_norm, conv in zip(self.bns_conv, self.convs):
            update = F.relu(
                conv(
                    batch_norm(hidden),
                    data.edge_index,
                    edge_weight=edge_weight,
                )
            )
            update = F.dropout(
                update,
                p=self.dropout,
                training=self.training,
            )
            hidden = hidden + update
        return hidden

    def __repr__(self):
        return self.__class__.__name__
