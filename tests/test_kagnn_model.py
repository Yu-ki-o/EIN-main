import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from model.KAGNN import KAGNN


def make_args(variant):
    return SimpleNamespace(
        max_hop=4,
        kagnn_variant=variant,
        kagnn_num_layers=2,
        kagnn_hidden_layers=1,
        kagnn_grid_size=3,
        kagnn_spline_order=2,
        kagnn_heads=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        classification_class_weights=[1.0, 1.0],
    )


def make_graph(label, num_nodes):
    edge_index = torch.stack(
        (
            torch.arange(0, num_nodes - 1),
            torch.arange(1, num_nodes),
        ),
        dim=0,
    )
    return Data(
        x=torch.randn(num_nodes, 5),
        edge_index=edge_index,
        y=torch.tensor([label]),
        user_state=torch.zeros(1, 4, 3),
    )


def make_batch():
    return Batch.from_data_list(
        [
            make_graph(0, 4),
            make_graph(1, 5),
        ]
    )


class KAGNNModelTest(unittest.TestCase):
    def test_variants_follow_ein_trainer_interface(self):
        for variant in [
            "KAGCN",
            "FASTKAGCN",
            "KAGAT",
            "FASTKAGAT",
            "KAGIN",
            "FASTKAGIN",
        ]:
            with self.subTest(variant=variant):
                args = make_args(variant)
                model = KAGNN(
                    in_feats=5,
                    hidden_dim=8,
                    num_classes=2,
                    args=args,
                    device=torch.device("cpu"),
                )
                data = make_batch()

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
                self.assertEqual(tuple(unknown.shape), (2, 4, 1))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
