import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)


class GCN_UncertaintySemanticChange(
    BiGCN_UncertaintySemanticChange
):
    """
    Plain-GCN backbone for uncertainty-routed semantic change modeling.

    This variant keeps the edge relation encoder, entropy-guided Binary
    Concrete sampling, support/deny routing, semantic-change encoder, trend
    branch, and configurable classification fusion from
    BiGCN_UncertaintySemanticChange.  Only the semantic-view encoder is
    replaced by a simple stacked GCN without BiGCN top-down/bottom-up fusion
    and without ResGCN residual updates.
    """

    backbone_type = "gcn"

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
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(in_feats, hid_feats))
        self.batch_norms.append(nn.BatchNorm1d(hid_feats))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hid_feats, hid_feats))
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
