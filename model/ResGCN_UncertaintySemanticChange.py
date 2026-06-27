import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d

from model.EIN_ResGCN import GCNConv

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)


class ResGCN_UncertaintySemanticChange(
    BiGCN_UncertaintySemanticChange
):
    """
    Residual-GCN backbone for uncertainty-routed semantic change modeling.

    This variant keeps the same edge relation, entropy, Binary Concrete
    sampling, semantic change, and trend branches as the BiGCN version, but
    each support/deny view is encoded once along the original propagation
    graph with stacked residual GCN layers. It does not concatenate separate
    top-down and bottom-up branch representations.
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
        self.bn_feat = BatchNorm1d(in_feats)
        self.conv_feat = GCNConv(in_feats, hid_feats, gfn=True)
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        edge_norm = bool(getattr(args, "edge_norm", True))
        for _ in range(max(1, int(getattr(args, "n_layers_conv", 3)))):
            self.bns_conv.append(BatchNorm1d(hid_feats))
            self.convs.append(
                GCNConv(
                    hid_feats,
                    hid_feats,
                    edge_norm=edge_norm,
                )
            )
        for module in self.modules():
            if isinstance(module, BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)

    def _encode_semantic_view(
        self,
        data,
        node_hidden,
        edge_index,
        edge_weight,
    ):
        hidden = self.bn_feat(data.x.float())
        hidden = F.relu(
            self.conv_feat(hidden, edge_index, edge_weight=edge_weight)
        )
        for batch_norm, conv in zip(self.bns_conv, self.convs):
            update = F.relu(
                conv(
                    batch_norm(hidden),
                    edge_index,
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
