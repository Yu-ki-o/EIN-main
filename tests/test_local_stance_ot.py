import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)
from model.local_stance_ot import LocalStanceOptimalTransportBranch


def make_args(mode):
    return SimpleNamespace(
        max_hop=4,
        dropout=0.0,
        global_pool="mean",
        n_layers_conv=2,
        relation_hidden_dim=8,
        use_trend_graph=False,
        use_semantic_tree_transformer="semantic_tree" in mode.lower(),
        semantic_tree_transformer_heads=2,
        semantic_tree_transformer_layers=1,
        semantic_tree_depth_dim=4,
        classification_fusion_mode=mode,
        classification_fusion_hidden_dim=16,
        ot_local_hops=3,
        ot_entropic_regularization=0.2,
        ot_sinkhorn_iterations=30,
        ot_structure_cost_weight=0.25,
        ot_max_events_per_side=8,
        lr=1e-3,
        weight_decay=0.0,
    )


def make_batch():
    mixed = Data(
        x=torch.randn(5, 5),
        # Root has one support and one deny reply.  The deeper labels stay
        # direct-parent events and are never XOR-composed by the OT branch.
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]]),
        edge_stance=torch.tensor([0, 1, 1, 0]),
        y=torch.tensor([1]),
        num_hop=torch.tensor([2]),
        user_state=torch.zeros(1, 4, 3),
    )
    one_sided = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_stance=torch.tensor([0, 0]),
        y=torch.tensor([0]),
        num_hop=torch.tensor([2]),
        user_state=torch.zeros(1, 4, 3),
    )
    return Batch.from_data_list([mixed, one_sided])


class LocalStanceOptimalTransportBranchTest(unittest.TestCase):
    def test_sinkhorn_matches_uniform_marginals(self):
        branch = LocalStanceOptimalTransportBranch(
            hidden_dim=8,
            max_hop=4,
            args=make_args("ot"),
        )
        cost = torch.tensor(
            [[0.1, 0.7, 0.4], [0.9, 0.2, 0.6]],
            dtype=torch.float32,
        )
        plan = branch._sinkhorn(cost)

        self.assertTrue(torch.isfinite(plan).all())
        self.assertTrue(
            torch.allclose(
                plan.sum(dim=1),
                torch.full((2,), 0.5),
                atol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                plan.sum(dim=0),
                torch.full((3,), 1.0 / 3.0),
                atol=1e-4,
            )
        )

    def test_capped_local_transport_is_edge_permutation_invariant(self):
        args = make_args("ot")
        args.ot_max_events_per_side = 4
        branch = LocalStanceOptimalTransportBranch(
            hidden_dim=8,
            max_hop=4,
            args=args,
        ).eval()
        node_hidden = torch.randn(41, 8)
        children = torch.arange(1, 41)
        edge_index = torch.stack((torch.zeros_like(children), children))
        edge_stance = torch.tensor([0] * 20 + [1] * 20)
        depth = torch.tensor([0] + [1] * 40)
        batch = torch.zeros(41, dtype=torch.long)
        roots = torch.tensor([0])

        first_graph, _, first_outputs = branch(
            node_hidden,
            edge_index,
            edge_stance,
            depth,
            batch,
            roots,
        )
        permutation = torch.randperm(edge_index.size(1))
        second_graph, _, second_outputs = branch(
            node_hidden,
            edge_index[:, permutation],
            edge_stance[permutation],
            depth,
            batch,
            roots,
        )

        self.assertTrue(torch.allclose(first_graph, second_graph, atol=1e-6))
        self.assertEqual(float(first_outputs["support_mass"][0]), 20.0)
        self.assertEqual(float(first_outputs["deny_mass"][0]), 20.0)
        self.assertEqual(
            float(first_outputs["selected_support_mass"][0]), 4.0
        )
        self.assertEqual(float(first_outputs["selected_deny_mass"][0]), 4.0)
        self.assertAlmostEqual(float(first_outputs["coverage"][0]), 0.2)
        self.assertTrue(
            torch.equal(
                first_outputs["transport_anchors"],
                second_outputs["transport_anchors"],
            )
        )

    def test_ot_mode_returns_node_and_graph_patterns_and_backpropagates(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args("OT"),
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(model.classification_branch_names, ("ot",))
        self.assertIsNone(model.semantic_tree_transformer)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(model._last_ot_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_ot_nodes.shape), (8, 8))
        self.assertTrue(torch.isfinite(model._last_ot_graph).all())
        self.assertTrue(model._last_ot_active_anchor[:5].any())
        self.assertEqual(float(model._last_ot_support_mass[0]), 2.0)
        self.assertEqual(float(model._last_ot_deny_mass[0]), 2.0)
        # Node 1's child directly denies node 1.  It stays a deny event rather
        # than being flipped by the root->node-1 support edge.
        self.assertEqual(float(model._last_ot_support_mass[1]), 0.0)
        self.assertEqual(float(model._last_ot_deny_mass[1]), 1.0)
        # The mixed graph has valid two-sided local fields, while the
        # all-support graph remains a finite one-sided representation.
        self.assertGreater(
            int(model._last_ot_transport_anchors.numel()),
            0,
        )
        self.assertIsNotNone(model.local_stance_ot.event_encoder[0].weight.grad)
        self.assertIsNotNone(model.local_stance_ot.ground_projection.weight.grad)

    def test_ot_semantic_tree_fuses_two_graph_branches(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args("ot_semantic_tree"),
            device=torch.device("cpu"),
        ).eval()

        output, _, _, _ = model(make_batch())

        self.assertEqual(
            model.classification_branch_names,
            ("ot", "semantic_tree"),
        )
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertEqual(tuple(model._last_ot_graph.shape), (2, 8))
        self.assertEqual(
            tuple(model._last_semantic_tree_graph.shape),
            (2, 8),
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_ot_mode_handles_a_single_node_graph(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args("ot"),
            device=torch.device("cpu"),
        ).eval()
        single = Data(
            x=torch.randn(1, 5),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_stance=torch.empty((0,), dtype=torch.long),
            y=torch.tensor([0]),
            num_hop=torch.tensor([1]),
            user_state=torch.zeros(1, 4, 3),
        )

        output, _, _, _ = model(Batch.from_data_list([single]))

        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(
            tuple(model._last_ot_transport_plans.shape),
            (0, 1, 1),
        )

    def test_legacy_mode_does_not_construct_ot_branch(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args("change"),
            device=torch.device("cpu"),
        ).eval()

        output, _, _, _ = model(make_batch())

        self.assertIsNone(model.local_stance_ot)
        self.assertIsNone(model._last_ot_graph)
        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
