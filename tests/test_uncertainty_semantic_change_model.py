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
from model.KAGNN_UncertaintySemanticChange import (
    KAGNN_UncertaintySemanticChange,
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
        use_ds_mass_routing=False,
        ds_unknown_prior=2.0,
        lambda_ds_unknown_edge_aux=0.0,
        use_global_ds_fusion=False,
        global_ds_unknown_prior=1.0,
        global_ds_temperature=1.0,
        global_ds_fusion_rule="dempster",
        global_ds_hidden_dim=8,
        use_degree_importance=True,
        degree_importance_strength=1.0,
        lambda_edge_relation_aux=0.1,
        lambda_view_mi_aux=0.0,
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

    def test_uncertainty_sampling_can_be_disabled(self):
        args = make_args()
        args.use_uncertainty_sampling = False
        router = EdgeRelationUncertaintyRouter(4, args).eval()
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
        self.assertAlmostEqual(float(keep), 1.0, places=6)
        self.assertAlmostEqual(float(support + deny), 1.0, places=6)

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

    def test_ds_mass_routing_keeps_unknown_mass_out_of_views(self):
        args = make_args()
        args.use_ds_mass_routing = True
        router = EdgeRelationUncertaintyRouter(4, args).eval()
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.zero_()
        nodes = torch.randn(2, 4)
        edge_index = torch.tensor([[0], [1]])

        logits, probabilities, unknown, keep, support, deny = router(
            nodes,
            edge_index,
        )
        masses, _, unknown_mass = router.relation_masses(logits)

        self.assertTrue(
            torch.allclose(
                masses.sum(dim=-1),
                torch.ones_like(unknown),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(unknown, unknown_mass, atol=1e-6))
        self.assertTrue(
            torch.allclose(probabilities, torch.tensor([[0.5, 0.5]]))
        )
        self.assertTrue(torch.allclose(keep, support + deny, atol=1e-6))
        self.assertTrue(torch.allclose(keep + unknown, torch.ones_like(keep)))
        self.assertGreater(float(unknown), float(support))


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
        self.assertIsNone(model.global_ds_fusion)
        self.assertIsNone(model._last_global_ds_masses)
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

    def test_dpga_semantic_change_encoder_forward(self):
        args = make_args()
        args.semantic_change_encoder = "dpga"
        args.dpga_pseudo_nodes = 3
        args.dpga_layers = 1
        args.dpga_attention_temperature = 1.0
        args.dpga_modulation_scale = 0.5
        args.dpga_use_node_weights = True
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, unknown, support, deny = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(model._last_change_nodes.shape), (5, 8))
        self.assertTrue(torch.isfinite(model._last_change_graph).all())
        self.assertIsNotNone(
            model.semantic_change_encoder.pseudo_nodes.grad
        )
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())

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

    def test_global_ds_fusion_outputs_mass_based_probabilities(self):
        args = make_args()
        args.use_global_ds_fusion = True
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = True
        args.classification_fusion_mode = "change_semantic_tree"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, unknown, support, deny = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(
            torch.allclose(
                output.exp().sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertEqual(tuple(model._last_global_ds_masses.shape), (2, 3))
        self.assertEqual(
            tuple(model._last_global_ds_branch_masses.shape),
            (2, 2, 3),
        )
        self.assertEqual(tuple(model._last_global_ds_conflict.shape), (2, 1))
        self.assertTrue(
            torch.allclose(
                model._last_global_ds_masses.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertIsNotNone(
            model.global_ds_fusion.mass_heads["change"][0].weight.grad
        )
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())

    def test_view_mi_auxiliary_loss_penalizes_correlated_views(self):
        args = make_args()
        args.lambda_view_mi_aux = 0.5
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        support_graph = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        deny_graph = support_graph.clone()

        loss = model._view_mutual_information_loss(
            support_graph,
            deny_graph,
        )
        self.assertGreater(float(loss), 0.0)

        model(make_batch())
        self.assertGreaterEqual(float(model._last_view_mi_loss), 0.0)
        self.assertTrue(
            torch.allclose(
                model.auxiliary_loss().detach(),
                model._last_edge_relation_loss + model._last_view_mi_loss,
                atol=1e-6,
            )
        )

    def test_vertical_path_attention_updates_only_non_root_nodes(self):
        args = make_args()
        args.use_vertical_path_attention = True
        args.classification_fusion_mode = "original_change_vertical"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        output, _, _, _ = model(data)
        roots = model._root_indices(data)
        non_root = torch.ones(data.x.size(0), dtype=torch.bool)
        non_root[roots.cpu()] = False
        node_hidden = model.node_projection(data.x.float())
        node_hidden = model._add_root_context(node_hidden, data)

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            model.classification_branch_names,
            ("original", "change", "vertical"),
        )
        self.assertEqual(model.fusion[0].in_features, 24)
        self.assertEqual(tuple(model._last_vertical_graph.shape), (2, 8))
        self.assertTrue(
            torch.allclose(
                model._last_vertical_nodes[roots],
                node_hidden[roots],
                atol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                model._last_vertical_nodes[non_root],
                node_hidden[non_root],
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_node_uncertainty[roots],
                torch.zeros_like(model._last_node_uncertainty[roots]),
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_node_uncertainty[
                    torch.tensor([1, 2, 4])
                ],
                model._last_edge_uncertainty,
            )
        )

    def test_semantic_tree_transformer_fuses_views_and_depth(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_vertical_path_attention = True
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = (
            "original_change_vertical_semantic_tree"
        )
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        output, _, _, _ = model(data)
        expected_depth = model._node_depths(data, data.edge_index)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            model.classification_branch_names,
            ("original", "change", "vertical", "semantic_tree"),
        )
        self.assertEqual(model.fusion[0].in_features, 32)
        self.assertEqual(
            tuple(model._last_semantic_tree_graph.shape),
            (2, 8),
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_nodes.shape),
            (5, 8),
        )
        self.assertTrue(
            torch.equal(model._last_semantic_tree_depth, expected_depth)
        )
        self.assertTrue(torch.isfinite(model._last_semantic_tree_graph).all())

    def test_change_semantic_tree_classification_fusion(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_vertical_path_attention = False
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = "change_semantic_tree"
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
            ("change", "semantic_tree"),
        )
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertIsNone(model._last_original_graph)
        self.assertIsNone(model._last_vertical_graph)
        self.assertEqual(tuple(model._last_change_graph.shape), (2, 8))
        self.assertEqual(
            tuple(model._last_semantic_tree_graph.shape),
            (2, 8),
        )

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

    def test_ds_mass_routing_forward_records_edge_masses(self):
        args = make_args()
        args.use_ds_mass_routing = True
        args.use_trend_graph = False
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, unknown, support, deny = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(model._last_edge_masses.shape), (3, 3))
        self.assertTrue(
            torch.allclose(
                model._last_edge_masses.sum(dim=-1),
                torch.ones(3),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_edge_unknown_mass,
                model._last_edge_uncertainty,
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())


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


class KAGNNUncertaintySemanticChangeTest(unittest.TestCase):
    def test_kagcn_variants_forward_use_weighted_semantic_views(self):
        for variant in ("KAGCN", "FASTKAGCN"):
            with self.subTest(variant=variant):
                args = make_args()
                args.kagnn_variant = variant
                args.kagnn_num_layers = 2
                args.kagnn_grid_size = 3
                args.kagnn_spline_order = 2
                args.use_vertical_path_attention = True
                args.vertical_path_attention_heads = 2
                args.classification_fusion_mode = "original_change_vertical"
                model = KAGNN_UncertaintySemanticChange(
                    in_feats=5,
                    hid_feats=8,
                    out_feats=8,
                    num_classes=2,
                    args=args,
                    device=torch.device("cpu"),
                ).train()
                data = make_batch()

                output, unknown, support, deny = model(data)
                loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
                loss.backward()

                self.assertEqual(tuple(output.shape), (2, 2))
                self.assertEqual(tuple(unknown.shape), (2, 4, 1))
                self.assertEqual(tuple(support.shape), (2, 4, 1))
                self.assertEqual(tuple(deny.shape), (2, 4, 1))
                self.assertEqual(len(model.kagnn_convs), 2)
                self.assertEqual(model.fusion[0].in_features, 24)
                self.assertEqual(
                    tuple(model._last_vertical_graph.shape),
                    (2, 8),
                )
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
