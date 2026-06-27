import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
    EdgeRelationUncertaintyRouter,
)
from model.ResGCN_UncertaintySemanticChange import (
    ResGCN_UncertaintySemanticChange,
)
from model.GCN_UncertaintySemanticChange import (
    GCN_UncertaintySemanticChange,
)
from model.GIN_UncertaintySemanticChange import (
    GIN_UncertaintySemanticChange,
)


def make_args():
    return SimpleNamespace(
        max_hop=4,
        dropout=0.0,
        global_pool="mean",
        n_layers_conv=2,
        relation_hidden_dim=8,
        relation_temperature=1.0,
        stance_route_temperature=0.5,
        stance_route_hard=True,
        uncertainty_sample_temperature=0.5,
        uncertainty_keep_floor=0.05,
        use_degree_importance=True,
        degree_importance_strength=1.0,
        lambda_edge_relation_aux=0.1,
        semantic_change_encoder="mlp",
        semantic_change_hidden_dim=8,
        uncertainty_trend_hidden_dim=8,
        use_trend_graph=True,
        use_node_keep_in_change_pool=True,
        classification_fusion_hidden_dim=16,
        lr=1e-3,
        weight_decay=0.0,
    )


def make_batch():
    first = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_stance=torch.tensor([0, 1]),
        y=torch.tensor([1]),
        num_hop=torch.tensor([2]),
        user_state=torch.zeros(1, 4, 3),
    )
    second = Data(
        x=torch.randn(2, 5),
        edge_index=torch.tensor([[0], [1]]),
        edge_stance=torch.tensor([0]),
        y=torch.tensor([0]),
        num_hop=torch.tensor([1]),
        user_state=torch.zeros(1, 4, 3),
    )
    return Batch.from_data_list([first, second])


class EdgeRelationUncertaintyRouterTest(unittest.TestCase):
    def test_equal_logits_have_maximum_entropy(self):
        router = EdgeRelationUncertaintyRouter(4, make_args())
        equal_prob = router.relation_probabilities(
            torch.tensor([[0.0, 0.0]])
        )
        confident_prob = router.relation_probabilities(
            torch.tensor([[8.0, -8.0]])
        )

        equal_entropy = router.normalized_entropy(equal_prob)
        confident_entropy = router.normalized_entropy(confident_prob)
        self.assertAlmostEqual(float(equal_entropy), 1.0, places=6)
        self.assertLess(float(confident_entropy), 1e-4)

    def test_eval_soft_sample_is_expected_keep_probability(self):
        router = EdgeRelationUncertaintyRouter(4, make_args()).eval()
        keep_probability = torch.tensor([0.1, 0.5, 0.9])
        sample = router.soft_bernoulli_sample(keep_probability)
        self.assertTrue(torch.allclose(sample, keep_probability))

    def test_train_soft_sample_is_differentiable(self):
        router = EdgeRelationUncertaintyRouter(4, make_args()).train()
        keep_probability = torch.tensor(
            [0.2, 0.8],
            requires_grad=True,
        )
        sample = router.soft_bernoulli_sample(keep_probability)
        sample.sum().backward()
        self.assertIsNotNone(keep_probability.grad)
        self.assertTrue(((sample > 0) & (sample < 1)).all())

    def test_low_degree_amplifies_existing_uncertainty(self):
        router = EdgeRelationUncertaintyRouter(4, make_args())
        uncertainty = torch.tensor([0.0, 0.5, 0.5])
        importance = torch.tensor([0.0, 1.0, 0.0])

        keep = router.reliability_probability(
            uncertainty,
            importance,
        )
        self.assertGreater(float(keep[0]), 0.999)
        self.assertGreater(float(keep[1]), float(keep[2]))

    def test_maximum_uncertainty_is_filtered_before_view_assignment(self):
        router = EdgeRelationUncertaintyRouter(4, make_args()).eval()
        router.set_epoch(router.warmup_epochs)
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.zero_()
        nodes = torch.randn(2, 4)
        edge_index = torch.tensor([[0], [1]])

        _, probabilities, uncertainty, keep, support, deny = router(
            nodes,
            edge_index,
        )
        self.assertTrue(
            torch.allclose(probabilities, torch.tensor([[0.5, 0.5]]))
        )
        self.assertAlmostEqual(float(uncertainty), 1.0, places=6)
        self.assertAlmostEqual(float(keep), router.keep_floor, places=6)
        self.assertAlmostEqual(
            float(support + deny),
            float(keep),
            places=6,
        )
        self.assertTrue(
            (float(support) == 0.0) ^ (float(deny) == 0.0)
        )

    def test_warmup_uses_soft_stance_route_without_edge_filtering(self):
        router = EdgeRelationUncertaintyRouter(4, make_args()).eval()
        router.set_epoch(0)
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.zero_()
        nodes = torch.randn(2, 4)
        edge_index = torch.tensor([[0], [1]])

        _, _, _, keep, support, deny = router(nodes, edge_index)
        self.assertAlmostEqual(float(keep), 1.0, places=6)
        self.assertAlmostEqual(float(support), 0.5, places=6)
        self.assertAlmostEqual(float(deny), 0.5, places=6)


class BiGCNUncertaintySemanticChangeTest(unittest.TestCase):
    def test_forward_outputs_all_framework_branches(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        output, unknown, support, deny = model(data)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(unknown.shape), (2, 4, 1))
        self.assertEqual(tuple(support.shape), (2, 4, 1))
        self.assertEqual(tuple(deny.shape), (2, 4, 1))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(
            tuple(model._last_change_nodes.shape),
            (5, 8),
        )
        self.assertEqual(
            tuple(model._last_original_graph.shape),
            (2, 8),
        )
        self.assertEqual(
            tuple(model._last_trend_sequence.shape),
            (2, 4, 5),
        )
        self.assertEqual(tuple(model._last_node_keep.shape), (5,))
        self.assertTrue(
            torch.allclose(
                model._last_child_degree_importance,
                torch.tensor([1.0, 0.0, 0.0]),
            )
        )
        self.assertEqual(
            model.fusion[0].in_features,
            model.hidden_dim * 3,
        )
        state_sequence = model._last_trend_sequence[:, :, :3]
        occupied_depth = state_sequence.sum(dim=-1) > 0
        occupied_mass = state_sequence.sum(dim=-1)[
            occupied_depth
        ]
        self.assertTrue(
            torch.allclose(
                occupied_mass,
                torch.ones_like(occupied_mass),
                atol=1e-6,
            )
        )

    def test_two_deny_edges_flip_state_back_to_support(self):
        args = make_args()
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        data = Batch.from_data_list(
            [
                Data(
                    x=torch.randn(3, 5),
                    edge_index=torch.tensor([[0, 1], [1, 2]]),
                    edge_stance=torch.tensor([1, 1]),
                    y=torch.tensor([1]),
                    num_hop=torch.tensor([2]),
                    user_state=torch.zeros(1, 4, 3),
                )
            ]
        )
        probabilities = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]]
        )
        keep = torch.ones(2)

        trend = model._build_uncertainty_trend(
            data,
            probabilities,
            keep,
        )
        self.assertTrue(
            torch.allclose(
                trend[0, 0, :3],
                torch.tensor([0.0, 0.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                trend[0, 1, :3],
                torch.tensor([1.0, 0.0, 0.0]),
            )
        )

    def test_trend_graph_can_be_excluded_from_classifier(self):
        args = make_args()
        args.use_trend_graph = False
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()

        output, unknown, support, deny = model(make_batch())
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertEqual(tuple(unknown.shape), (2, 4, 1))
        self.assertEqual(tuple(support.shape), (2, 4, 1))
        self.assertEqual(tuple(deny.shape), (2, 4, 1))

    def test_support_deny_change_classification_fusion(self):
        args = make_args()
        args.classification_fusion_mode = "support_deny_change"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()

        output, _, _, _ = model(make_batch())
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            model.classification_branch_names,
            ("support", "deny", "change"),
        )
        self.assertEqual(model.fusion[0].in_features, 24)
        self.assertIsNone(model._last_original_graph)
        self.assertEqual(tuple(model._last_support_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_deny_graph.shape), (2, 8))

    def test_view_pooling_uses_incoming_semantic_edge_weight(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).eval()
        data = Batch.from_data_list(
            [
                Data(
                    x=torch.randn(3, 5),
                    edge_index=torch.tensor([[0, 1], [1, 2]]),
                    edge_stance=torch.tensor([0, 1]),
                    y=torch.tensor([0]),
                    num_hop=torch.tensor([2]),
                    user_state=torch.zeros(1, 4, 3),
                )
            ]
        )

        support_node_weight = model._build_view_node_weight(
            data,
            torch.tensor([1.0, 0.0]),
        )
        deny_node_weight = model._build_view_node_weight(
            data,
            torch.tensor([0.0, 1.0]),
        )
        self.assertTrue(
            torch.equal(
                support_node_weight,
                torch.tensor([1.0, 1.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                deny_node_weight,
                torch.tensor([1.0, 0.0, 1.0]),
            )
        )

    def test_change_pool_can_skip_second_node_keep_weighting(self):
        args = make_args()
        args.use_node_keep_in_change_pool = False
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        model(data)
        expected_change_graph = model.global_pool(
            model._last_change_nodes,
            data.batch,
        )
        self.assertTrue(
            torch.allclose(
                model._last_change_graph,
                expected_change_graph,
                atol=1e-6,
            )
        )

    def test_removed_parent_disconnects_descendant_from_root(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).eval()
        data = Batch.from_data_list(
            [
                Data(
                    x=torch.randn(3, 5),
                    edge_index=torch.tensor([[0, 1], [1, 2]]),
                    edge_stance=torch.tensor([0, 0]),
                    y=torch.tensor([0]),
                    num_hop=torch.tensor([2]),
                    user_state=torch.zeros(1, 4, 3),
                )
            ]
        )
        keep = torch.tensor([0.0, 1.0])
        node_keep = model._build_root_connected_keep(data, keep)
        self.assertTrue(
            torch.equal(node_keep, torch.tensor([1.0, 0.0, 0.0]))
        )

    def test_classification_and_edge_losses_backpropagate(self):
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, unknown, support, deny = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNotNone(model.edge_router.logit_head.weight.grad)
        self.assertEqual(
            float(model.physics_loss(unknown, support, deny, data.user_state)),
            0.0,
        )


class ResGCNUncertaintySemanticChangeTest(unittest.TestCase):
    def test_resgcn_forward_uses_single_residual_direction(self):
        model = ResGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        output, unknown, support, deny = model(data)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(unknown.shape), (2, 4, 1))
        self.assertEqual(tuple(support.shape), (2, 4, 1))
        self.assertEqual(tuple(deny.shape), (2, 4, 1))
        self.assertEqual(model.fusion[0].in_features, 24)
        self.assertEqual(tuple(model._last_original_graph.shape), (2, 8))


class GCNUncertaintySemanticChangeTest(unittest.TestCase):
    def test_plain_gcn_forward_uses_single_stacked_gcn_view(self):
        model = GCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        output, unknown, support, deny = model(data)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(unknown.shape), (2, 4, 1))
        self.assertEqual(tuple(support.shape), (2, 4, 1))
        self.assertEqual(tuple(deny.shape), (2, 4, 1))
        self.assertEqual(model.fusion[0].in_features, 24)
        self.assertEqual(tuple(model._last_original_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_support_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_deny_graph.shape), (2, 8))
        self.assertEqual(len(model.convs), make_args().n_layers_conv)


class GINUncertaintySemanticChangeTest(unittest.TestCase):
    def test_plain_gin_forward_uses_single_stacked_gin_view(self):
        model = GIN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        output, unknown, support, deny = model(data)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(unknown.shape), (2, 4, 1))
        self.assertEqual(tuple(support.shape), (2, 4, 1))
        self.assertEqual(tuple(deny.shape), (2, 4, 1))
        self.assertEqual(model.fusion[0].in_features, 24)
        self.assertEqual(tuple(model._last_original_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_support_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_deny_graph.shape), (2, 8))
        self.assertEqual(len(model.convs), make_args().n_layers_conv)


if __name__ == "__main__":
    unittest.main()
