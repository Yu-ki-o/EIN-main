import torch.nn.functional as F
from torch import nn

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)
from model.KAGNN import FastKAGCNLayer, KAGCNLayer


class KAGNN_UncertaintySemanticChange(BiGCN_UncertaintySemanticChange):
    """
    KAGNN-backed uncertainty-routed semantic change model.

    The uncertainty router, support/deny graph construction, semantic-change
    branch, vertical path attention, and classifier fusion are inherited from
    BiGCN_UncertaintySemanticChange. Only the semantic-view GNN encoder is
    replaced with KAGCN-style KAN graph convolution layers.
    """

    backbone_type = "kagnn"

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
        variant = str(getattr(args, "kagnn_variant", "KAGCN")).upper()
        self.kagnn_variant = variant
        self.kagnn_num_layers = max(
            1,
            int(
                getattr(
                    args,
                    "kagnn_num_layers",
                    getattr(args, "n_layers_conv", 2),
                )
            ),
        )
        grid_size = int(getattr(args, "kagnn_grid_size", 4))
        spline_order = int(getattr(args, "kagnn_spline_order", 3))

        if variant == "KAGCN":
            layer_factory = lambda in_dim, out_dim: KAGCNLayer(
                in_dim,
                out_dim,
                grid_size=grid_size,
                spline_order=spline_order,
            )
        elif variant in {"FASTKAGCN", "FKAGCN"}:
            layer_factory = lambda in_dim, out_dim: FastKAGCNLayer(
                in_dim,
                out_dim,
                grid_size=grid_size,
            )
        else:
            raise ValueError(
                "KAGNN_UncertaintySemanticChange currently supports "
                "KAGCN/FASTKAGCN because these layers accept edge_weight; "
                "got {}".format(variant)
            )

        self.kagnn_convs = nn.ModuleList()
        self.kagnn_convs.append(layer_factory(in_feats, hid_feats))
        for _ in range(self.kagnn_num_layers - 1):
            self.kagnn_convs.append(layer_factory(hid_feats, hid_feats))

    def _encode_semantic_view(
        self,
        data,
        node_hidden,
        edge_index,
        edge_weight,
    ):
        hidden = data.x.float()
        for conv in self.kagnn_convs:
            hidden = conv(
                hidden,
                edge_index,
                edge_weight=edge_weight,
            )
            hidden = F.silu(hidden)
            hidden = F.dropout(
                hidden,
                p=self.dropout,
                training=self.training,
            )
        return hidden

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.kagnn_variant)
