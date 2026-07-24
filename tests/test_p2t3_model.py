import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from model.P2T3 import P2T3
from utils.p2t3 import (
    attach_p2t3_sequence_metadata,
    build_p2t3_sequence_metadata,
)


def make_args(unsup_weight=0.0):
    return SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        max_hop=6,
        base_model="P2T3",
        p2t3_d_model=8,
        p2t3_num_layers=2,
        p2t3_num_heads=2,
        p2t3_dim_feedforward=16,
        p2t3_dropout=0.0,
        p2t3_max_sequence_length=32,
        p2t3_max_chain_length=8,
        p2t3_max_chain_identifiers=8,
        p2t3_max_depth=8,
        p2t3_use_chain_identifier=True,
        p2t3_use_depth_embedding=True,
        p2t3_use_type_embedding=True,
        p2t3_measure="JSD",
        p2t3_unsup_weight=unsup_weight,
        classification_class_weights=[1.0, 1.0],
    )


def make_graph(label, with_metadata):
    # Chains: 0-1-3, 0-1-4-5, and the shallow conversation 0-2.
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 4],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    data = Data(
        x=torch.randn(6, 5),
        edge_index=edge_index,
        directed_edge_index=edge_index.clone(),
        y=torch.tensor([label]),
        user_state=torch.zeros(1, 6, 3),
    )
    if with_metadata:
        attach_p2t3_sequence_metadata(data, make_args())
    return data


class P2T3SequenceMetadataTest(unittest.TestCase):
    def test_conversation_chains_repeat_shared_prefixes(self):
        edge_index = torch.tensor(
            [
                [0, 0, 1, 1, 4],
                [1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        metadata = build_p2t3_sequence_metadata(
            edge_index,
            num_nodes=6,
            max_sequence_length=32,
            max_chain_length=8,
            max_chain_identifiers=8,
        )

        self.assertEqual(
            metadata["p2t3_node_id"].tolist(),
            [0, 1, 3, 1, 4, 5, 2],
        )
        self.assertEqual(
            metadata["p2t3_chain_id"].tolist(),
            [0, 1, 1, 2, 2, 2, 3],
        )
        self.assertEqual(
            metadata["p2t3_depth"].tolist(),
            [0, 1, 2, 1, 2, 3, 1],
        )
        self.assertEqual(
            metadata["p2t3_type_id"].tolist(),
            [0, 1, 1, 1, 1, 1, 2],
        )
        self.assertEqual(
            metadata["p2t3_level_one_mask"].tolist(),
            [False, True, False, True, False, False, True],
        )

    def test_all_deep_chains_precede_shallow_conversations(self):
        # Root child 1 is shallow while the later root child 2 is deep.
        edge_index = torch.tensor(
            [
                [0, 0, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        )
        metadata = build_p2t3_sequence_metadata(
            edge_index,
            num_nodes=4,
            max_sequence_length=16,
            max_chain_length=8,
            max_chain_identifiers=8,
        )
        self.assertEqual(
            metadata["p2t3_node_id"].tolist(),
            [0, 2, 3, 1],
        )
        self.assertEqual(
            metadata["p2t3_type_id"].tolist(),
            [0, 1, 1, 2],
        )

    def test_sequence_length_is_a_hard_limit(self):
        edge_index = torch.tensor(
            [
                [0, 0, 1, 1, 4],
                [1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        metadata = build_p2t3_sequence_metadata(
            edge_index,
            num_nodes=6,
            max_sequence_length=4,
            max_chain_length=8,
            max_chain_identifiers=8,
        )
        self.assertEqual(metadata["p2t3_sequence_length"].item(), 4)
        self.assertEqual(metadata["p2t3_node_id"].tolist(), [0, 1, 3, 1])


class P2T3ModelTest(unittest.TestCase):
    def test_forward_matches_ein_trainer_interface(self):
        torch.manual_seed(3)
        model = P2T3(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        )
        data = Batch.from_data_list(
            [
                make_graph(0, with_metadata=True),
                make_graph(1, with_metadata=True),
            ]
        )

        output, unknown, support, deny = model(data)
        loss = model.classification_loss(output, data.y)
        loss = loss + model.physics_loss(
            unknown,
            support,
            deny,
            data.user_state,
        )
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(unknown.shape), (2, 6, 1))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(model.lin_class.weight.grad)
        self.assertEqual(model._last_sequence_lengths.tolist(), [7, 7])

    def test_on_the_fly_metadata_matches_cached_metadata(self):
        torch.manual_seed(7)
        args = make_args()
        model = P2T3(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        graph = make_graph(0, with_metadata=False)
        cached_graph = graph.clone()
        attach_p2t3_sequence_metadata(cached_graph, args)

        dynamic_output = model(Batch.from_data_list([graph]))[0]
        cached_output = model(Batch.from_data_list([cached_graph]))[0]
        self.assertTrue(
            torch.allclose(dynamic_output, cached_output, atol=1e-6)
        )

    def test_optional_mi_loss_backpropagates(self):
        torch.manual_seed(11)
        model = P2T3(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(unsup_weight=0.1),
            device=torch.device("cpu"),
        ).train()
        data = Batch.from_data_list(
            [
                make_graph(0, with_metadata=True),
                make_graph(1, with_metadata=True),
            ]
        )

        output, _, _, _ = model(data)
        auxiliary = model.auxiliary_loss()
        loss = model.classification_loss(output, data.y) + auxiliary
        loss.backward()

        self.assertTrue(torch.isfinite(auxiliary))
        self.assertIsNotNone(model.local_d.linear_shortcut.weight.grad)
        self.assertIsNotNone(model.global_d.linear_shortcut.weight.grad)

    def test_native_pretraining_loss_updates_input_projection(self):
        torch.manual_seed(13)
        model = P2T3(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).train()
        data = Batch.from_data_list(
            [
                make_graph(0, with_metadata=True),
                make_graph(1, with_metadata=True),
            ]
        )

        loss = model.pretraining_loss(data)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNone(model.lin_class.weight.grad)

    def test_native_checkpoint_loads_projection_and_chain_basis_only(self):
        args = make_args()
        source = P2T3(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            source.input_projection.weight.fill_(0.25)
            source.chain_identifier_matrix.copy_(torch.eye(8))

        target = P2T3(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        )
        target_classifier = target.lin_class.weight.detach().clone()

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "pretrained.pt"
            torch.save({"state_dict": source.state_dict()}, checkpoint)
            result = target.load_pretrained(checkpoint, map_location="cpu")

        self.assertEqual(
            result.missing_keys,
            ["lin_class.weight", "lin_class.bias"],
        )
        self.assertTrue(
            torch.allclose(
                target.input_projection.weight,
                source.input_projection.weight,
            )
        )
        self.assertTrue(
            torch.allclose(
                target.chain_identifier_matrix,
                source.chain_identifier_matrix,
            )
        )
        self.assertTrue(
            torch.allclose(target.lin_class.weight, target_classifier)
        )


if __name__ == "__main__":
    unittest.main()
