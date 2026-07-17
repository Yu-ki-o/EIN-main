import torch
import torch.nn.functional as F

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)
from model.ResGCN_StaticDynamicSemanticChange import (
    ResGCN_StaticDynamicSemanticChange,
)


class BiGCN_StaticDynamicSemanticChange(
    ResGCN_StaticDynamicSemanticChange
):
    """BiGCN counterpart of the isolated static/dynamic change model.

    The input/root-context representation is the initial pseudo-time state.
    Every subsequent state is the fusion of top-down and bottom-up outputs at
    the same BiGCN layer. The final state exactly matches the dimensional
    contract of the original BiGCN semantic-view encoder.
    """

    def _build_view_backbone(
        self,
        in_feats,
        hid_feats,
        out_feats,
        args,
    ):
        BiGCN_UncertaintySemanticChange._build_view_backbone(
            self,
            in_feats,
            hid_feats,
            out_feats,
            args,
        )

    def _encode_bigcn_direction_trajectory(
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
        root_hidden = self._extend_root_features(hidden_first, data)

        # A first-layer state is retained for temporal modeling. The root
        # channel follows the original BiGCN representation contract.
        history = [
            torch.cat((F.relu(hidden_first), root_hidden), dim=-1)
        ]

        root_raw = self._extend_root_features(raw_nodes, data)
        hidden = torch.cat((hidden_first, root_raw), dim=-1)
        hidden = F.relu(hidden)
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        hidden = conv2(
            hidden,
            edge_index,
            edge_weight=edge_weight,
        )
        hidden = F.relu(hidden)
        history.append(torch.cat((hidden, root_hidden), dim=-1))

        for conv in extra_convs or []:
            root_current = self._extend_root_features(hidden, data)
            hidden = torch.cat((hidden, root_current), dim=-1)
            hidden = F.relu(hidden)
            hidden = F.dropout(
                hidden,
                p=self.dropout,
                training=self.training,
            )
            hidden = conv(
                hidden,
                edge_index,
                edge_weight=edge_weight,
            )
            hidden = F.relu(hidden)
            history.append(torch.cat((hidden, root_hidden), dim=-1))
        return history

    def _encode_bigcn_view_trajectory(
        self,
        data,
        edge_index,
        edge_weight,
    ):
        top_down = self._encode_bigcn_direction_trajectory(
            data,
            edge_index,
            edge_weight,
            self.td_conv1,
            self.td_conv2,
            self.td_extra_convs,
        )
        bottom_up = self._encode_bigcn_direction_trajectory(
            data,
            self._reverse_edges(edge_index),
            edge_weight,
            self.bu_conv1,
            self.bu_conv2,
            self.bu_extra_convs,
        )
        if len(top_down) != len(bottom_up):
            raise RuntimeError("top-down and bottom-up trajectories must align")
        return [
            self.direction_fusion(torch.cat((td_state, bu_state), dim=-1))
            for td_state, bu_state in zip(top_down, bottom_up)
        ]

    def _encode_semantic_views(
        self,
        data,
        node_hidden,
        edge_index,
        support_weight,
        deny_weight,
    ):
        support_history = [node_hidden]
        support_history.extend(
            self._encode_bigcn_view_trajectory(
                data,
                edge_index,
                support_weight,
            )
        )
        deny_history = [node_hidden]
        deny_history.extend(
            self._encode_bigcn_view_trajectory(
                data,
                edge_index,
                deny_weight,
            )
        )
        original_weight = support_weight.new_ones(support_weight.shape)
        original_history = [node_hidden]
        original_history.extend(
            self._encode_bigcn_view_trajectory(
                data,
                edge_index,
                original_weight,
            )
        )

        self.static_dynamic_encoder.set_trajectories(
            support_history,
            deny_history,
            original_history,
            self._node_depths(data, edge_index),
        )
        return support_history[-1], deny_history[-1]

    def __repr__(self):
        return self.__class__.__name__
