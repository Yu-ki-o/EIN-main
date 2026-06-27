import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import MessagePassing

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)


class WeightedGINConv(MessagePassing):
    """
    GIN layer that keeps the model's support/deny edge routing weights.
    """

    def __init__(self, in_channels, out_channels, train_eps=False):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight.view(-1),
        )
        return self.mlp((1.0 + self.eps) * x + out)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GIN_UncertaintySemanticChange(BiGCN_UncertaintySemanticChange):
    """
    Plain-GIN backbone for uncertainty-routed semantic change modeling.

    This variant keeps the edge relation encoder, entropy-guided Binary
    Concrete sampling, support/deny routing, semantic-change encoder, trend
    branch, and configurable classification fusion from
    BiGCN_UncertaintySemanticChange. Only the semantic-view encoder is
    replaced by stacked weighted GIN layers.
    """

    backbone_type = "gin"

    def __init__(
        self,
        in_feats,
        hid_feats,
        out_feats,
        num_classes,
        args,
        device,
    ):
        super().__init__(
            in_feats=in_feats,
            hid_feats=hid_feats,
            out_feats=out_feats,
            num_classes=num_classes,
            args=args,
            device=device,
        )

    def _build_view_backbone(
        self,
        in_feats,
        hid_feats,
        out_feats,
        args,
    ):
        num_layers = max(1, int(getattr(args, "n_layers_conv", 2)))
        train_eps = bool(getattr(args, "gin_train_eps", False))
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(
            WeightedGINConv(
                in_feats,
                hid_feats,
                train_eps=train_eps,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hid_feats))
        for _ in range(num_layers - 1):
            self.convs.append(
                WeightedGINConv(
                    hid_feats,
                    hid_feats,
                    train_eps=train_eps,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hid_feats))

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)

    def _encode_semantic_view(
        self,
        data,
        node_hidden,
        edge_index,
        edge_weight,
    ):
        hidden = data.x.float()
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            hidden = conv(
                hidden,
                edge_index,
                edge_weight=edge_weight,
            )
            hidden = batch_norm(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(
                hidden,
                p=self.dropout,
                training=self.training,
            )
        return hidden

    def __repr__(self):
        return self.__class__.__name__
