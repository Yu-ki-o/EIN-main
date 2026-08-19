import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
    EdgeRelationUncertaintyRouter,
    SemanticParityDirectionEncoder,
    SemanticParityGCNDirectionEncoder,
    SemanticParityProbabilisticSGCNDirectionEncoder,
    SemanticTreeTransformerBranch,
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
from model.collective_revision import CollectiveRevisionEncoder
from model.conflict_hotspot_field import ConflictHotspotField
from model.semantic_tree_attention_complementary_fusion import (
    SemanticTreeAttentionComplementaryFusion,
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
        edge_relation_distribution=None,
        use_ds_mass_routing=False,
        use_dirichlet_relation_routing=False,
        dirichlet_relation_prior=1.0,
        dirichlet_relation_sample=False,
        dirichlet_teacher_strength=10.0,
        dirichlet_teacher_smoothing=0.05,
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
        use_structural_balance_loss=False,
        lambda_structural_balance_aux=0.05,
        structural_balance_warmup_epochs=5,
        lambda_view_mi_aux=0.0,
        use_semantic_parity_gnn=True,
        semantic_parity_aggregation="mean",
        semantic_parity_residual=True,
        semantic_node_weight_mode="local",
        semantic_change_encoder="mlp",
        semantic_change_hidden_dim=8,
        use_gaussian_semantic_change_bottleneck=False,
        semantic_change_gaussian_latent_dim=8,
        semantic_change_gaussian_sample=False,
        semantic_change_gaussian_min_logvar=-8.0,
        semantic_change_gaussian_max_logvar=4.0,
        lambda_semantic_change_bottleneck=0.0,
        lambda_semantic_tree_change_mi_aux=0.0,
        use_semantic_tree_change_uncertainty_bias=False,
        semantic_tree_uncertainty_source="gaussian_change",
        semantic_tree_uncertainty_bias_scale=1.0,
        semantic_tree_change_uncertainty_detach=True,
        use_semantic_tree_reliability_hinge_loss=False,
        semantic_tree_reliability_hinge_margin=0.1,
        lambda_semantic_tree_reliability_hinge_aux=0.05,
        use_change_uncertainty_pooling=False,
        change_uncertainty_pool_scale=1.0,
        change_uncertainty_pool_detach=True,
        semantic_tree_input_mode="support_deny",
        semantic_tree_query_mode="root_learned",
        semantic_tree_num_queries=1,
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


class SemanticTreeAttentionComplementaryFusionTest(unittest.TestCase):
    def test_conditional_redundancy_updates_tree_but_not_change(self):
        args = make_args()
        args.semantic_tree_conditional_redundancy_stop_change = True
        module = SemanticTreeAttentionComplementaryFusion(
            8,
            2,
            args=args,
        ).train()
        base_fused = torch.randn(4, 8)
        change_graph = torch.randn(4, 8, requires_grad=True)
        value_dense = torch.randn(4, 3, 8, requires_grad=True)
        attention = torch.softmax(torch.randn(4, 1, 3), dim=-1)
        valid_mask = torch.ones(4, 3, dtype=torch.bool)
        target = torch.tensor([0, 0, 1, 1])

        fused = module(
            base_fused,
            change_graph,
            value_dense,
            attention,
            valid_mask,
            target=target,
        )
        redundancy = module.last_conditional_redundancy_loss
        redundancy.backward()

        self.assertTrue(torch.equal(fused, base_fused))
        self.assertGreater(float(redundancy), 0.0)
        self.assertIsNotNone(value_dense.grad)
        self.assertIsNone(change_graph.grad)


class ConflictHotspotFieldTest(unittest.TestCase):
    def test_diffusion_and_coverage_are_differentiable(self):
        args = SimpleNamespace(
            conflict_hotspot_diffusion_steps=2,
            conflict_hotspot_diffusion_alpha=0.4,
            conflict_hotspot_diffusion_direction="undirected",
            conflict_hotspot_use_change_pooling=True,
            conflict_hotspot_change_pool_scale=1.0,
            conflict_hotspot_use_semantic_tree_bias=True,
            conflict_hotspot_semantic_tree_bias_scale=0.5,
            lambda_conflict_hotspot_coverage_aux=0.1,
            conflict_hotspot_coverage_temperature=0.5,
            conflict_hotspot_coverage_detach=True,
            conflict_hotspot_dropout=0.0,
        )
        module = ConflictHotspotField(8, args=args).train()
        change_nodes = torch.randn(5, 8, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
        batch = torch.tensor([0, 0, 0, 1, 1])

        outputs = module(change_nodes, edge_index, batch)
        attention = torch.softmax(torch.randn(2, 1, 3), dim=-1)
        valid = torch.tensor([[True, True, True], [True, True, False]])
        coverage = module.coverage_loss(
            outputs["normalized_field"],
            batch,
            attention,
            valid,
        )
        loss = outputs["field_intensity"].mean() + coverage
        loss.backward()

        self.assertEqual(tuple(outputs["field_intensity"].shape), (5,))
        self.assertTrue((outputs["pool_multiplier"] >= 1.0).all())
        self.assertTrue(torch.isfinite(coverage))
        self.assertIsNotNone(change_nodes.grad)
        self.assertIsNotNone(module.intensity[-1].weight.grad)


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

    def test_ucst_moves_ambiguous_reversal_toward_half(self):
        args = make_args()
        args.use_ucst = True
        args.ucst_reversal_temperature = 1.0
        args.ucst_uncertainty_strength = 1.0
        args.ucst_reliability_strength = 0.75
        router = EdgeRelationUncertaintyRouter(4, args).eval()
        logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        uncertainty = torch.tensor([1.0, 0.0])
        edge_features = torch.zeros(2, 16)

        delta, reversal, calibrated, reliability = (
            router.ucst_transport_parameters(
                logits,
                uncertainty,
                edge_features,
            )
        )
        expected_reversal = torch.sigmoid(torch.tensor(2.0))

        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(
            torch.allclose(
                reversal,
                expected_reversal.expand_as(reversal),
            )
        )
        self.assertAlmostEqual(float(calibrated[0]), 0.5, places=6)
        self.assertAlmostEqual(
            float(calibrated[1]),
            float(expected_reversal),
            places=6,
        )
        self.assertTrue(
            torch.allclose(reliability, torch.tensor([0.25, 1.0]))
        )

        preserve, reverse = router.ucst_transport_weights(
            calibrated,
            reliability,
            torch.ones(2),
        )
        self.assertTrue(torch.allclose(preserve + reverse, reliability))

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
        router.set_epoch(router.warmup_epochs)
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

    def test_ds_uncertainty_can_be_reserved_for_semantic_attention(self):
        args = make_args()
        args.use_ds_mass_routing = True
        args.use_uncertainty_sampling = False
        router = EdgeRelationUncertaintyRouter(4, args).eval()
        router.set_epoch(router.warmup_epochs)
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.zero_()
        nodes = torch.randn(2, 4)
        edge_index = torch.tensor([[0], [1]])

        _, probabilities, unknown, keep, support, deny = router(
            nodes,
            edge_index,
        )

        self.assertGreater(float(unknown), 0.0)
        self.assertTrue(
            torch.allclose(probabilities, torch.tensor([[0.5, 0.5]]))
        )
        self.assertAlmostEqual(float(keep), 1.0, places=6)
        self.assertAlmostEqual(float(support + deny), 1.0, places=6)

    def test_ds_and_dirichlet_relation_routing_are_mutually_exclusive(self):
        args = make_args()
        args.use_ds_mass_routing = True
        args.use_dirichlet_relation_routing = True

        with self.assertRaises(ValueError):
            EdgeRelationUncertaintyRouter(4, args)

    def test_distribution_switch_closes_other_relation_modes(self):
        args = make_args()
        args.edge_relation_distribution = "dirichlet"
        args.use_ds_mass_routing = True
        args.use_dirichlet_relation_routing = False

        router = EdgeRelationUncertaintyRouter(4, args)

        self.assertTrue(router.use_dirichlet_relation_routing)
        self.assertFalse(router.use_ds_mass_routing)
        self.assertEqual(router.edge_relation_distribution, "dirichlet")

    def test_dirichlet_relation_routing_uses_mean_probability(self):
        args = make_args()
        args.use_dirichlet_relation_routing = True
        router = EdgeRelationUncertaintyRouter(4, args).eval()
        logits = torch.tensor([[0.0, 0.0], [4.0, -4.0]])

        concentration, probabilities = router.dirichlet_relation_probabilities(
            logits
        )
        expected = concentration / concentration.sum(dim=-1, keepdim=True)

        self.assertTrue((concentration > 0).all())
        self.assertTrue(torch.allclose(probabilities, expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(float(probabilities[0, 0]), 0.5, places=6)
        self.assertGreater(float(probabilities[1, 0]), float(probabilities[1, 1]))


class SemanticParityDirectionEncoderTest(unittest.TestCase):
    def _identity_encoder(self, num_layers):
        encoder = SemanticParityDirectionEncoder(
            input_dim=2,
            hidden_dim=2,
            num_layers=num_layers,
            dropout=0.0,
            residual=False,
        ).eval()
        with torch.no_grad():
            encoder.input_projection.weight.copy_(torch.eye(2))
            encoder.input_projection.bias.zero_()
            for layer in encoder.layers:
                layer.weight.copy_(torch.eye(2))
            for norm in encoder.norms:
                norm.weight.fill_(1.0)
                norm.bias.zero_()
        return encoder

    def _root_channels_for_path(self, signs):
        num_layers = len(signs)
        encoder = self._identity_encoder(num_layers)
        num_nodes = num_layers + 1
        x = torch.zeros(num_nodes, 2)
        x[-1, 0] = 1.0
        edge_index = torch.tensor(
            [
                list(range(num_layers, 0, -1)),
                list(range(num_layers - 1, -1, -1)),
            ],
            dtype=torch.long,
        )
        support_weight = torch.tensor(
            [1.0 if sign == "S" else 0.0 for sign in signs]
        )
        deny_weight = 1.0 - support_weight
        support_nodes, deny_nodes = encoder(
            x,
            edge_index,
            support_weight,
            deny_weight,
        )
        return support_nodes[0].abs().sum(), deny_nodes[0].abs().sum()

    def test_path_parity_is_valid_for_one_to_four_layers(self):
        cases = [
            ("S", "support"),
            ("D", "deny"),
            ("DD", "support"),
            ("SSD", "deny"),
            ("DSD", "support"),
            ("SDSD", "support"),
            ("SSSD", "deny"),
        ]
        for signs, expected in cases:
            with self.subTest(signs=signs):
                support_mass, deny_mass = self._root_channels_for_path(signs)
                if expected == "support":
                    self.assertGreater(float(support_mass), 0.0)
                    self.assertEqual(float(deny_mass), 0.0)
                else:
                    self.assertEqual(float(support_mass), 0.0)
                    self.assertGreater(float(deny_mass), 0.0)


class SemanticParityGCNDirectionEncoderTest(unittest.TestCase):
    def _identity_encoder(self, num_layers):
        encoder = SemanticParityGCNDirectionEncoder(
            input_dim=2,
            hidden_dim=2,
            num_layers=num_layers,
            dropout=0.0,
            residual=False,
        ).eval()
        with torch.no_grad():
            encoder.input_projection.weight.copy_(torch.eye(2))
            encoder.input_projection.bias.zero_()
            for layer in encoder.layers:
                layer.lin.weight.copy_(torch.eye(2))
            for norm in encoder.norms:
                norm.weight.fill_(1.0)
                norm.bias.zero_()
        return encoder

    def _root_channels_for_path(self, signs):
        num_layers = len(signs)
        encoder = self._identity_encoder(num_layers)
        num_nodes = num_layers + 1
        x = torch.zeros(num_nodes, 2)
        x[-1, 0] = 1.0
        edge_index = torch.tensor(
            [
                list(range(num_layers, 0, -1)),
                list(range(num_layers - 1, -1, -1)),
            ],
            dtype=torch.long,
        )
        support_weight = torch.tensor(
            [1.0 if sign == "S" else 0.0 for sign in signs]
        )
        deny_weight = 1.0 - support_weight
        support_nodes, deny_nodes = encoder(
            x,
            edge_index,
            support_weight,
            deny_weight,
        )
        return support_nodes[0].abs().sum(), deny_nodes[0].abs().sum()

    def test_gcn_path_parity_is_valid_for_one_to_four_layers(self):
        cases = [
            ("S", "support"),
            ("D", "deny"),
            ("DD", "support"),
            ("SSD", "deny"),
            ("DSD", "support"),
            ("SDSD", "support"),
            ("SSSD", "deny"),
        ]
        for signs, expected in cases:
            with self.subTest(signs=signs):
                support_mass, deny_mass = self._root_channels_for_path(signs)
                if expected == "support":
                    self.assertGreater(float(support_mass), 0.0)
                    self.assertEqual(float(deny_mass), 0.0)
                else:
                    self.assertEqual(float(support_mass), 0.0)
                    self.assertGreater(float(deny_mass), 0.0)

    def test_model_selects_gcn_parity_aggregation(self):
        args = make_args()
        args.semantic_parity_aggregation = "gcn"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()

        output, unknown, support, deny = model(make_batch())

        self.assertEqual(model.semantic_parity_encoder.aggregation, "gcn")
        self.assertIsInstance(
            model.semantic_parity_encoder.top_down,
            SemanticParityGCNDirectionEncoder,
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())


class SemanticParityProbabilisticSGCNDirectionEncoderTest(unittest.TestCase):
    def _neighbor_only_encoder(self, num_layers):
        encoder = SemanticParityProbabilisticSGCNDirectionEncoder(
            input_dim=2,
            hidden_dim=2,
            num_layers=num_layers,
            dropout=0.0,
            residual=False,
        ).eval()
        with torch.no_grad():
            encoder.input_projection.weight.copy_(torch.eye(2))
            encoder.input_projection.bias.zero_()
            for layer_index, (support_layer, deny_layer) in enumerate(
                zip(encoder.support_layers, encoder.deny_layers)
            ):
                support_layer.weight.zero_()
                deny_layer.weight.zero_()
                support_layer.weight[:, :2].copy_(torch.eye(2))
                deny_layer.weight[:, :2].copy_(torch.eye(2))
                if layer_index > 0:
                    support_layer.weight[:, 2:4].copy_(torch.eye(2))
                    deny_layer.weight[:, 2:4].copy_(torch.eye(2))
            for norm in (*encoder.support_norms, *encoder.deny_norms):
                norm.weight.fill_(1.0)
                norm.bias.zero_()
        return encoder

    def _root_channels_for_path(self, signs):
        num_layers = len(signs)
        encoder = self._neighbor_only_encoder(num_layers)
        node_features = torch.zeros(num_layers + 1, 2)
        node_features[-1] = torch.tensor([1.0, -1.0])
        edge_index = torch.tensor(
            [
                list(range(num_layers, 0, -1)),
                list(range(num_layers - 1, -1, -1)),
            ],
            dtype=torch.long,
        )
        support_weight = torch.tensor(
            [1.0 if sign == "S" else 0.0 for sign in signs]
        )
        deny_weight = 1.0 - support_weight
        support_nodes, deny_nodes = encoder(
            node_features,
            edge_index,
            support_weight,
            deny_weight,
        )
        return support_nodes[0].abs().sum(), deny_nodes[0].abs().sum()

    def test_binary_probabilities_recover_sgcn_path_parity(self):
        cases = [
            ("S", "support"),
            ("D", "deny"),
            ("DD", "support"),
            ("SSD", "deny"),
            ("DSD", "support"),
            ("SDSD", "support"),
            ("SSSD", "deny"),
        ]
        for signs, expected in cases:
            with self.subTest(signs=signs):
                support_mass, deny_mass = self._root_channels_for_path(signs)
                if expected == "support":
                    self.assertGreater(float(support_mass), 0.0)
                    self.assertEqual(float(deny_mass), 0.0)
                else:
                    self.assertEqual(float(support_mass), 0.0)
                    self.assertGreater(float(deny_mass), 0.0)

    def test_probability_mass_mean_softly_splits_an_edge(self):
        encoder = self._neighbor_only_encoder(num_layers=1)
        node_features = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
        edge_index = torch.tensor([[0], [1]])

        support_neighbor = encoder._probability_mass_mean(
            node_features,
            edge_index,
            torch.tensor([0.7]),
        )
        deny_neighbor = encoder._probability_mass_mean(
            node_features,
            edge_index,
            torch.tensor([0.3]),
        )

        self.assertTrue(
            torch.allclose(
                support_neighbor[1],
                torch.tensor([1.4, -1.4]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                deny_neighbor[1],
                torch.tensor([0.6, -0.6]),
                atol=1e-6,
            )
        )

    def test_soft_probabilities_update_both_channels_and_are_differentiable(self):
        torch.manual_seed(23)
        encoder = SemanticParityProbabilisticSGCNDirectionEncoder(
            input_dim=3,
            hidden_dim=4,
            num_layers=2,
            dropout=0.0,
            residual=True,
        ).eval()
        node_features = torch.randn(2, 3)
        edge_index = torch.tensor([[0], [1]])
        support_weight = torch.tensor([0.7], requires_grad=True)
        deny_weight = torch.tensor([0.3], requires_grad=True)

        support_nodes, deny_nodes = encoder(
            node_features,
            edge_index,
            support_weight,
            deny_weight,
        )
        (support_nodes[1].sum() + deny_nodes[1].sum()).backward()

        self.assertGreater(float(support_nodes[1].abs().sum()), 0.0)
        self.assertGreater(float(deny_nodes[1].abs().sum()), 0.0)
        self.assertIsNotNone(support_weight.grad)
        self.assertIsNotNone(deny_weight.grad)
        self.assertTrue(torch.isfinite(support_weight.grad).all())
        self.assertTrue(torch.isfinite(deny_weight.grad).all())

    def test_model_selects_probability_sgcn_and_keeps_soft_edge_routes(self):
        args = make_args()
        args.semantic_parity_aggregation = "probabilistic_sgcn"
        args.use_uncertainty_sampling = False
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        model.set_epoch(model.edge_router.warmup_epochs)

        output, unknown, support, deny = model(make_batch())

        self.assertEqual(
            model.semantic_parity_encoder.aggregation,
            "probabilistic_sgcn",
        )
        self.assertIsInstance(
            model.semantic_parity_encoder.top_down,
            SemanticParityProbabilisticSGCNDirectionEncoder,
        )
        self.assertTrue(
            torch.allclose(
                model._last_support_weight,
                model._last_edge_probabilities[:, 0],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_deny_weight,
                model._last_edge_probabilities[:, 1],
                atol=1e-6,
            )
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())

    def test_probability_sgcn_requires_semantic_parity_encoder(self):
        args = make_args()
        args.semantic_parity_aggregation = "probabilistic_sgcn"
        args.use_semantic_parity_gnn = False

        with self.assertRaisesRegex(
            ValueError,
            "requires use_semantic_parity_gnn",
        ):
            BiGCN_UncertaintySemanticChange(
                in_feats=5,
                hid_feats=8,
                out_feats=8,
                num_classes=2,
                args=args,
                device=torch.device("cpu"),
            )


class BiGCNUncertaintySemanticChangeTest(unittest.TestCase):
    def test_conflict_hotspot_disabled_preserves_legacy_parameterization(self):
        legacy_args = make_args()
        legacy_args.use_trend_graph = False
        legacy_args.classification_fusion_mode = "change_semantic_tree"
        explicit_args = make_args()
        explicit_args.use_trend_graph = False
        explicit_args.classification_fusion_mode = "change_semantic_tree"
        explicit_args.use_conflict_hotspot_field = False

        torch.manual_seed(307)
        legacy_model = BiGCN_UncertaintySemanticChange(
            5, 8, 8, 2, legacy_args, torch.device("cpu")
        ).eval()
        torch.manual_seed(307)
        explicit_model = BiGCN_UncertaintySemanticChange(
            5, 8, 8, 2, explicit_args, torch.device("cpu")
        ).eval()

        data = make_batch()
        legacy_output = legacy_model(data)
        explicit_output = explicit_model(data)
        self.assertEqual(
            tuple(legacy_model.state_dict().keys()),
            tuple(explicit_model.state_dict().keys()),
        )
        for legacy_value, explicit_value in zip(legacy_output, explicit_output):
            self.assertTrue(torch.equal(legacy_value, explicit_value))
        self.assertIsNone(explicit_model.conflict_hotspot_field)

    def test_conflict_hotspot_field_guides_change_and_tree(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change_semantic_tree"
        args.semantic_tree_depth_dim = 4
        args.use_conflict_hotspot_field = True
        args.conflict_hotspot_diffusion_steps = 2
        args.conflict_hotspot_diffusion_alpha = 0.35
        args.conflict_hotspot_use_change_pooling = True
        args.conflict_hotspot_use_semantic_tree_bias = True
        args.lambda_conflict_hotspot_coverage_aux = 0.05
        model = BiGCN_UncertaintySemanticChange(
            5, 8, 8, 2, args, torch.device("cpu")
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            tuple(model._last_conflict_hotspot_field.shape),
            (5,),
        )
        self.assertEqual(
            tuple(model._last_conflict_hotspot_attention_bias.shape),
            (5,),
        )
        self.assertTrue(
            torch.isfinite(model._last_conflict_hotspot_coverage_loss)
        )
        self.assertIsNotNone(
            model.conflict_hotspot_field.intensity[-1].weight.grad
        )

    def test_original_semantic_tree_fusion_uses_two_graph_views(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = False
        args.classification_fusion_mode = "original_semantic_tree"
        args.semantic_tree_input_mode = "original"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()

        with torch.no_grad():
            output, _, _, _ = model(make_batch())

        self.assertEqual(
            model.classification_branch_names,
            ("original", "semantic_tree"),
        )
        self.assertIsNotNone(model.semantic_tree_transformer)
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIsNotNone(model._last_original_graph)
        self.assertIsNotNone(model._last_semantic_tree_graph)
        self.assertIsNone(model._last_change_graph)

    def test_structural_balance_disabled_preserves_legacy_behavior(self):
        legacy_args = make_args()
        del legacy_args.use_structural_balance_loss
        explicit_args = make_args()

        torch.manual_seed(103)
        legacy_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=legacy_args,
            device=torch.device("cpu"),
        ).eval()
        torch.manual_seed(103)
        explicit_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=explicit_args,
            device=torch.device("cpu"),
        ).eval()
        data = make_batch()

        legacy_output = legacy_model(data)
        explicit_output = explicit_model(data)

        self.assertFalse(legacy_model.use_structural_balance_loss)
        self.assertEqual(
            tuple(legacy_model.state_dict().keys()),
            tuple(explicit_model.state_dict().keys()),
        )
        for name, legacy_value in legacy_model.state_dict().items():
            self.assertTrue(
                torch.equal(legacy_value, explicit_model.state_dict()[name]),
                msg=name,
            )
        for legacy_value, explicit_value in zip(
            legacy_output,
            explicit_output,
        ):
            self.assertTrue(torch.equal(legacy_value, explicit_value))
        self.assertTrue(
            torch.equal(
                legacy_model.auxiliary_loss(),
                explicit_model.auxiliary_loss(),
            )
        )
        self.assertEqual(
            float(legacy_model._last_structural_balance_loss),
            0.0,
        )
        self.assertIsNone(
            legacy_model._last_structural_balance_pair_index
        )

    def test_structural_balance_reuses_relation_head_for_two_hop_paths(self):
        args = make_args()
        args.use_structural_balance_loss = True
        args.lambda_structural_balance_aux = 0.2
        args.structural_balance_warmup_epochs = 0
        args.lambda_edge_relation_aux = 0.0
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

        expected_pair_index = torch.tensor([[0], [2]])
        expected_target = torch.tensor([1])
        self.assertTrue(
            torch.equal(
                model._last_structural_balance_pair_index,
                expected_pair_index,
            )
        )
        self.assertTrue(
            torch.equal(
                model._last_structural_balance_target,
                expected_target,
            )
        )
        node_hidden = model.node_projection(data.x.float())
        node_hidden = model._add_root_context(node_hidden, data)
        pair_logits, _, _ = model.edge_router.relation_outputs(
            node_hidden,
            expected_pair_index,
        )
        expected_loss = 0.2 * F.cross_entropy(
            pair_logits,
            expected_target,
        )
        self.assertTrue(
            torch.allclose(
                model._last_structural_balance_loss,
                expected_loss,
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.auxiliary_loss().detach(),
                model._last_structural_balance_loss,
                atol=1e-7,
            )
        )

        model.zero_grad()
        model(data)
        model.auxiliary_loss().backward()
        relation_grad = model.edge_router.logit_head.weight.grad
        self.assertIsNotNone(relation_grad)
        self.assertGreater(float(relation_grad.abs().sum()), 0.0)

    def test_structural_balance_handles_undirected_edges_and_warmup(self):
        args = make_args()
        args.use_structural_balance_loss = True
        args.lambda_structural_balance_aux = 0.1
        args.structural_balance_warmup_epochs = 2
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
                    edge_index=torch.tensor(
                        [[0, 1, 1, 2], [1, 2, 0, 1]]
                    ),
                    edge_stance=torch.tensor([0, 1, 0, 1]),
                    y=torch.tensor([1]),
                    num_hop=torch.tensor([2]),
                    user_state=torch.zeros(1, 4, 3),
                )
            ]
        )

        model.set_epoch(1)
        model(data)
        self.assertEqual(
            float(model._last_structural_balance_loss),
            0.0,
        )
        self.assertIsNone(model._last_structural_balance_pair_index)

        model.set_epoch(2)
        model(data)
        self.assertTrue(
            torch.equal(
                model._last_structural_balance_pair_index,
                torch.tensor([[0], [2]]),
            )
        )
        self.assertTrue(
            torch.equal(
                model._last_structural_balance_target,
                torch.tensor([1]),
            )
        )
        self.assertGreater(
            float(model._last_structural_balance_loss),
            0.0,
        )

    def test_omitted_parity_aggregation_is_exactly_explicit_mean(self):
        default_args = make_args()
        del default_args.semantic_parity_aggregation
        explicit_args = make_args()

        torch.manual_seed(101)
        default_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=default_args,
            device=torch.device("cpu"),
        ).eval()
        torch.manual_seed(101)
        explicit_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=explicit_args,
            device=torch.device("cpu"),
        ).eval()
        batch = make_batch()

        default_output = default_model(batch)
        explicit_output = explicit_model(batch)

        self.assertEqual(default_model.semantic_parity_aggregation, "mean")
        self.assertEqual(
            tuple(default_model.state_dict().keys()),
            tuple(explicit_model.state_dict().keys()),
        )
        for name, default_value in default_model.state_dict().items():
            self.assertTrue(
                torch.equal(default_value, explicit_model.state_dict()[name]),
                msg=name,
            )
        for default_value, explicit_value in zip(
            default_output,
            explicit_output,
        ):
            self.assertTrue(torch.equal(default_value, explicit_value))

    def test_cest_disabled_preserves_legacy_parameterization(self):
        legacy_args = make_args()
        legacy_args.use_trend_graph = False
        legacy_args.classification_fusion_mode = "change_semantic_tree"
        explicit_args = make_args()
        explicit_args.use_trend_graph = False
        explicit_args.classification_fusion_mode = "change_semantic_tree"
        explicit_args.use_cross_scale_evidence_transition = False

        torch.manual_seed(83)
        legacy_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=legacy_args,
            device=torch.device("cpu"),
        ).eval()
        torch.manual_seed(83)
        explicit_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=explicit_args,
            device=torch.device("cpu"),
        ).eval()

        legacy_state = legacy_model.state_dict()
        explicit_state = explicit_model.state_dict()
        self.assertEqual(set(legacy_state), set(explicit_state))
        for name in legacy_state:
            self.assertTrue(
                torch.equal(legacy_state[name], explicit_state[name]),
                msg=name,
            )
        data = make_batch()
        legacy_output, _, _, _ = legacy_model(data)
        explicit_output, _, _, _ = explicit_model(data)
        self.assertTrue(
            torch.allclose(legacy_output, explicit_output, atol=1e-7)
        )
        self.assertIsNone(legacy_model.cross_scale_evidence_transition)

    def test_cest_refines_change_and_tree_with_state_transitions(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_cross_scale_evidence_transition = True
        args.cest_hidden_dim = 8
        args.cest_use_relation_condition = True
        args.cest_use_semantic_residual = True
        args.cest_use_depth_phase = False
        args.cest_use_uncertainty_weight = False
        args.cest_residual_gate_init = -2.0
        args.cest_dropout = 0.0
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        F.nll_loss(output, data.y).backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(model._last_cest_node_states.shape), (5, 4))
        self.assertTrue(
            torch.allclose(
                model._last_cest_node_states.sum(dim=-1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        self.assertEqual(
            tuple(model._last_cest_transition_profile.shape),
            (2, 1, 2, 4, 4),
        )
        self.assertTrue(
            torch.allclose(
                model._last_cest_transition_profile.flatten(1).sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertEqual(tuple(model._last_cest_pattern_graph.shape), (2, 8))
        self.assertTrue(torch.isfinite(model._last_cest_pattern_graph).all())
        self.assertIsNotNone(
            model.cross_scale_evidence_transition.change_score[1].weight.grad
        )
        self.assertIsNotNone(
            model.cross_scale_evidence_transition.tree_score[1].weight.grad
        )

    def test_cest_requires_both_change_and_semantic_tree_branches(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change"
        args.use_cross_scale_evidence_transition = True

        with self.assertRaisesRegex(
            ValueError,
            "containing both 'change' and 'semantic_tree'",
        ):
            BiGCN_UncertaintySemanticChange(
                in_feats=5,
                hid_feats=8,
                out_feats=8,
                num_classes=2,
                args=args,
                device=torch.device("cpu"),
            )

    def test_disabled_attention_complementary_fusion_preserves_parameters(self):
        legacy_args = make_args()
        legacy_args.use_trend_graph = False
        legacy_args.classification_fusion_mode = "change_semantic_tree"
        explicit_args = make_args()
        explicit_args.use_trend_graph = False
        explicit_args.classification_fusion_mode = "change_semantic_tree"
        explicit_args.use_semantic_tree_attention_complementary_fusion = False

        torch.manual_seed(37)
        legacy_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=legacy_args,
            device=torch.device("cpu"),
        ).eval()
        torch.manual_seed(37)
        explicit_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=explicit_args,
            device=torch.device("cpu"),
        ).eval()

        legacy_state = legacy_model.state_dict()
        explicit_state = explicit_model.state_dict()
        self.assertEqual(set(legacy_state), set(explicit_state))
        for name in legacy_state:
            self.assertTrue(
                torch.equal(legacy_state[name], explicit_state[name]),
                msg=name,
            )
        self.assertIsNone(
            legacy_model.semantic_tree_attention_complementary_fusion
        )

    def test_attention_complementary_fusion_runs_and_backpropagates(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change_semantic_tree"
        args.semantic_tree_query_mode = "learned"
        args.semantic_tree_depth_dim = 4
        args.use_semantic_tree_attention_complementary_fusion = True
        args.semantic_tree_evidence_temperature = 0.5
        args.semantic_tree_evidence_rank_margin = 0.2
        args.semantic_tree_evidence_residual_init = 0.0
        args.semantic_tree_evidence_warmup_epochs = 2
        args.semantic_tree_conditional_redundancy_stop_change = True
        args.lambda_semantic_tree_evidence_sufficiency_aux = 0.01
        args.lambda_semantic_tree_evidence_rank_aux = 0.005
        args.lambda_semantic_tree_conditional_redundancy_aux = 0.001
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        model.set_epoch(1)
        data = make_batch()

        output, _, _, _ = model(data)
        loss = model.classification_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        module = model.semantic_tree_attention_complementary_fusion
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            tuple(model._last_semantic_tree_high_evidence.shape),
            (2, 8),
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_low_evidence.shape),
            (2, 8),
        )
        self.assertTrue(
            torch.allclose(
                model._last_semantic_tree_evidence_high_weight.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_semantic_tree_evidence_low_weight.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.isfinite(
                model._last_semantic_tree_evidence_sufficiency_loss
            )
        )
        self.assertTrue(
            torch.isfinite(model._last_semantic_tree_evidence_rank_loss)
        )
        self.assertTrue(
            torch.isfinite(
                model._last_semantic_tree_conditional_redundancy_loss
            )
        )
        self.assertIsNotNone(module.residual_scale.grad)
        self.assertIsNotNone(module.evidence_classifier.weight.grad)

    def test_attention_complementary_zero_residual_matches_old_logits(self):
        base_args = make_args()
        base_args.use_trend_graph = False
        base_args.classification_fusion_mode = "change_semantic_tree"
        enabled_args = make_args()
        enabled_args.use_trend_graph = False
        enabled_args.classification_fusion_mode = "change_semantic_tree"
        enabled_args.use_semantic_tree_attention_complementary_fusion = True
        enabled_args.semantic_tree_evidence_residual_init = 0.0
        enabled_args.lambda_semantic_tree_evidence_sufficiency_aux = 0.0
        enabled_args.lambda_semantic_tree_evidence_rank_aux = 0.0
        enabled_args.lambda_semantic_tree_conditional_redundancy_aux = 0.0

        torch.manual_seed(41)
        base_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=base_args,
            device=torch.device("cpu"),
        ).eval()
        torch.manual_seed(41)
        enabled_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=enabled_args,
            device=torch.device("cpu"),
        ).eval()
        enabled_state = enabled_model.state_dict()
        enabled_state.update(base_model.state_dict())
        enabled_model.load_state_dict(enabled_state)
        data = make_batch()

        base_output, _, _, _ = base_model(data)
        enabled_output, _, _, _ = enabled_model(data)

        self.assertTrue(
            torch.allclose(base_output, enabled_output, atol=1e-7)
        )

    def test_ucst_disabled_preserves_legacy_parameterization(self):
        legacy_args = make_args()
        explicit_args = make_args()
        explicit_args.use_ucst = False

        torch.manual_seed(13)
        legacy_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=legacy_args,
            device=torch.device("cpu"),
        ).eval()
        torch.manual_seed(13)
        explicit_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=explicit_args,
            device=torch.device("cpu"),
        ).eval()

        legacy_state = legacy_model.state_dict()
        explicit_state = explicit_model.state_dict()
        self.assertEqual(set(legacy_state), set(explicit_state))
        for name in legacy_state:
            self.assertTrue(
                torch.equal(legacy_state[name], explicit_state[name]),
                msg=name,
            )
        self.assertIsNone(legacy_model.edge_router.reversal_delta_head)

    def test_ucst_enabled_runs_continuous_transport_and_backpropagates(self):
        args = make_args()
        args.use_ucst = True
        args.use_uncertainty_sampling = False
        args.ucst_reversal_temperature = 1.0
        args.ucst_uncertainty_strength = 1.0
        args.ucst_reliability_strength = 0.5
        args.ucst_detach_uncertainty = True
        args.ucst_delta_warmup_epochs = 0
        args.lambda_ucst_delta_aux = 1e-3
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        num_edges = data.edge_index.size(1)
        self.assertEqual(
            tuple(model._last_ucst_reversal_strength.shape),
            (num_edges,),
        )
        self.assertTrue(
            ((model._last_ucst_calibrated_reversal >= 0.0)
             & (model._last_ucst_calibrated_reversal <= 1.0)).all()
        )
        self.assertTrue(
            ((model._last_ucst_reliability >= 0.0)
             & (model._last_ucst_reliability <= 1.0)).all()
        )
        self.assertTrue(
            torch.allclose(
                model._last_support_weight + model._last_deny_weight,
                model._last_ucst_reliability,
                atol=1e-6,
            )
        )
        self.assertIsNotNone(
            model.edge_router.reversal_delta_head[-1].weight.grad
        )
        self.assertTrue(torch.isfinite(model.auxiliary_loss()))

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

    def test_conflict_transformer_builds_interpretable_head_biases(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_conflict_field_bottleneck = True
        args.conflict_encoder_mode = "transformer"
        args.conflict_attention_heads = 4
        args.conflict_attention_layers = 1
        args.conflict_attention_ffn_dim = 16
        args.conflict_attention_depth_dim = 4
        args.conflict_attention_max_depth = 4
        args.conflict_attention_dropout = 0.0
        args.classification_fusion_mode = "conflict"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        encoder = model.conflict_field_bottleneck.attention_encoder
        score = torch.tensor([[0.1, 0.4, 0.9]])
        valid = torch.ones(1, 3, dtype=torch.bool)

        bias = encoder.build_attention_bias(score, valid).view(
            1,
            4,
            3,
            3,
        )

        self.assertEqual(
            encoder.head_roles,
            ("key", "boundary", "region", "free"),
        )
        self.assertGreater(float(bias[0, 0, 0, 2]), float(bias[0, 0, 0, 0]))
        self.assertGreater(float(bias[0, 1, 0, 2]), float(bias[0, 1, 0, 0]))
        self.assertGreater(float(bias[0, 2, 2, 2]), float(bias[0, 2, 0, 0]))
        self.assertTrue(torch.equal(bias[0, 3], torch.zeros(3, 3)))

    def test_conflict_transformer_classifies_without_change_encoder(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_conflict_field_bottleneck = True
        args.conflict_encoder_mode = "transformer"
        args.conflict_attention_heads = 4
        args.conflict_attention_layers = 1
        args.conflict_attention_ffn_dim = 16
        args.conflict_attention_depth_dim = 4
        args.conflict_attention_max_depth = 4
        args.conflict_attention_dropout = 0.0
        args.conflict_attention_pool = "mean"
        args.classification_fusion_mode = "conflict"
        args.lambda_edge_relation_aux = 0.0
        args.lambda_view_mi_aux = 0.0
        args.lambda_conflict_label_aux = 0.0
        args.lambda_conflict_size_aux = 0.0
        args.lambda_conflict_redundancy_aux = 0.0
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        F.nll_loss(output, data.y).backward()

        self.assertEqual(model.classification_branch_names, ("conflict",))
        self.assertIsNone(model._last_change_nodes)
        self.assertIsNone(model._last_change_graph)
        self.assertEqual(tuple(model._last_conflict_graph.shape), (2, 8))
        self.assertEqual(tuple(model._last_conflict_nodes.shape), (5, 8))
        self.assertEqual(
            tuple(model._last_conflict_attention_received.shape),
            (5,),
        )
        self.assertEqual(
            tuple(model._last_conflict_attention_by_head.shape),
            (5, 4),
        )
        self.assertEqual(
            tuple(model._last_conflict_node_importance.shape),
            (5,),
        )
        self.assertTrue(
            torch.allclose(
                model._last_conflict_node_importance,
                model._last_conflict_attention_received
                * model._last_conflict_score,
                atol=1e-6,
            )
        )
        attention_encoder = (
            model.conflict_field_bottleneck.attention_encoder
        )
        self.assertIsNotNone(
            attention_encoder.blocks[0].self_attn.in_proj_weight.grad
        )
        self.assertIsNotNone(attention_encoder.key_scale_raw.grad)
        self.assertIsNone(
            model.semantic_change_encoder.encoder[0].weight.grad
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

    def test_collective_revision_branch_can_be_enabled_by_fusion_mode(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change_collective_revision"
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
            ("change", "collective_revision"),
        )
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertEqual(
            tuple(model._last_collective_revision_graph.shape),
            (2, 8),
        )
        self.assertEqual(
            tuple(
                model._last_collective_revision_outputs["sequence"].shape
            ),
            (2, 4, 5),
        )

    def test_collective_revision_distinguishes_success_from_resistance(self):
        args = make_args()
        args.collective_revision_window_k = 1
        args.collective_revision_threshold_learnable = False
        args.collective_revision_adoption_threshold_init = 0.01
        args.collective_revision_challenge_threshold_init = 0.01
        args.collective_revision_gate_temperature = 0.02
        args.collective_revision_min_gain = 0.05
        encoder = CollectiveRevisionEncoder(8, args).eval()

        current = [0.75, 0.0, 0.25, 1.0, 0.5]
        resistant_future = [0.90, 0.0, 0.10, 1.0, 0.0]
        successful_future = [0.40, 0.0, 0.60, 1.0, 0.0]
        padding = [0.0, 0.0, 0.0, 0.0, 0.0]
        resistant_trend = torch.tensor(
            [[current, resistant_future, padding]],
            dtype=torch.float32,
        )
        successful_trend = torch.tensor(
            [[current, successful_future, padding]],
            dtype=torch.float32,
        )

        _, resistant = encoder(resistant_trend, torch.tensor([2]))
        _, successful = encoder(successful_trend, torch.tensor([2]))

        self.assertGreater(
            float(resistant["revision_resistance"][0, 0]),
            float(successful["revision_resistance"][0, 0]),
        )
        self.assertGreater(
            float(successful["revision_success"][0, 0]),
            float(resistant["revision_success"][0, 0]),
        )

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
        self.assertEqual(
            tuple(model._last_original_nodes.shape),
            (5, 8),
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_topics.shape),
            (2, 3, 1),
        )
        self.assertIsNone(model._last_semantic_tree_topic_similarity)
        self.assertEqual(
            tuple(model._last_semantic_tree_query.shape),
            (2, 1, 8),
        )
        self.assertTrue(
            torch.allclose(
                model._last_semantic_tree_attention[0].sum(dim=-1),
                torch.ones(1),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_semantic_tree_attention[1].sum(dim=-1),
                torch.ones(1),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(model._last_semantic_tree_graph).all())

    def test_disabled_semantic_tree_experts_preserve_legacy_parameters(self):
        legacy_args = make_args()
        legacy_args.semantic_tree_depth_dim = 4
        explicit_args = make_args()
        explicit_args.semantic_tree_depth_dim = 4
        explicit_args.use_semantic_tree_query_experts = False

        torch.manual_seed(29)
        legacy_branch = SemanticTreeTransformerBranch(
            8,
            args=legacy_args,
            num_classes=2,
        )
        torch.manual_seed(29)
        explicit_branch = SemanticTreeTransformerBranch(
            8,
            args=explicit_args,
            num_classes=2,
        )

        legacy_state = legacy_branch.state_dict()
        explicit_state = explicit_branch.state_dict()
        self.assertEqual(set(legacy_state), set(explicit_state))
        for name in legacy_state:
            self.assertTrue(
                torch.equal(legacy_state[name], explicit_state[name]),
                msg=name,
            )
        self.assertIsNone(legacy_branch.query_experts)

    def test_low_rank_semantic_tree_experts_route_and_backpropagate(self):
        args = make_args()
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_query_mode = "learned"
        args.use_semantic_tree_query_experts = True
        args.semantic_tree_query_expert_num = 4
        args.semantic_tree_query_expert_topk = 2
        args.semantic_tree_query_expert_basis_rank = 3
        args.semantic_tree_query_expert_adapter_rank = 3
        args.semantic_tree_query_expert_warmup_epochs = 0
        branch = SemanticTreeTransformerBranch(
            8,
            args=args,
            num_classes=2,
        ).train()
        branch.set_epoch(10)
        original = torch.randn(7, 8)
        support = torch.randn(7, 8)
        deny = torch.randn(7, 8)
        depth = torch.tensor([0, 1, 2, 0, 1, 1, 2])
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        target = torch.tensor([0, 1])

        graph, nodes = branch(
            original,
            support,
            deny,
            depth,
            batch,
            target=target,
        )
        auxiliary = (
            branch.last_expert_classification_loss
            + branch.last_expert_routing_loss
            + branch.last_expert_diversity_loss
            + branch.last_expert_counterfactual_loss
        )
        (graph.pow(2).mean() + auxiliary).backward()

        self.assertEqual(tuple(graph.shape), (2, 8))
        self.assertEqual(tuple(nodes.shape), (7, 8))
        self.assertEqual(tuple(branch.last_expert_queries.shape), (2, 4, 8))
        self.assertEqual(
            tuple(branch.last_expert_router_probability.shape),
            (2, 4),
        )
        self.assertTrue(
            torch.allclose(
                branch.last_expert_route_weight.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertTrue(
            branch.last_expert_route_weight.gt(0.0).sum(dim=-1).eq(2).all()
        )
        self.assertTrue(
            torch.allclose(
                branch.last_expert_responsibility.sum(dim=-1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertIsNotNone(branch.query_experts.query_basis.grad)
        self.assertIsNotNone(branch.query_experts.router.weight.grad)
        self.assertTrue(torch.isfinite(auxiliary))

    def test_semantic_tree_expert_losses_enter_model_auxiliary_loss(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = True
        args.classification_fusion_mode = "change_semantic_tree"
        args.semantic_tree_query_mode = "learned"
        args.semantic_tree_depth_dim = 4
        args.use_semantic_tree_query_experts = True
        args.semantic_tree_query_expert_num = 4
        args.semantic_tree_query_expert_topk = 2
        args.semantic_tree_query_expert_basis_rank = 3
        args.semantic_tree_query_expert_adapter_rank = 3
        args.semantic_tree_query_expert_warmup_epochs = 0
        args.lambda_semantic_tree_query_expert_classification_aux = 0.01
        args.lambda_semantic_tree_query_expert_routing_aux = 0.01
        args.lambda_semantic_tree_query_expert_diversity_aux = 0.001
        args.lambda_semantic_tree_query_expert_counterfactual_aux = 0.01
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        model.set_epoch(10)
        data = make_batch()

        output, _, _, _ = model(data)
        loss = model.classification_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            tuple(model._last_semantic_tree_expert_queries.shape),
            (2, 4, 8),
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_expert_route_weight.shape),
            (2, 4),
        )
        self.assertTrue(
            torch.isfinite(
                model._last_semantic_tree_query_expert_classification_loss
            )
        )
        self.assertTrue(
            torch.isfinite(
                model._last_semantic_tree_query_expert_counterfactual_loss
            )
        )
        self.assertIsNotNone(
            model.semantic_tree_transformer.query_experts.query_basis.grad
        )

    def test_semantic_tree_depth_is_part_of_dual_view_values(self):
        args = make_args()
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        branch = SemanticTreeTransformerBranch(8, args=args).eval()
        original = torch.randn(3, 8)
        support = torch.randn(3, 8)
        deny = torch.randn(3, 8)
        batch = torch.zeros(3, dtype=torch.long)

        _, shallow_nodes = branch(
            original,
            support,
            deny,
            torch.tensor([0, 1, 1]),
            batch,
        )
        shallow_key = branch.last_key.clone()
        shallow_value = branch.last_value.clone()
        _, deep_nodes = branch(
            original,
            support,
            deny,
            torch.tensor([0, 2, 2]),
            batch,
        )

        self.assertTrue(torch.allclose(shallow_key, branch.last_key))
        self.assertFalse(torch.allclose(shallow_value, branch.last_value))
        self.assertFalse(torch.allclose(shallow_nodes, deep_nodes))

    def test_gaussian_shared_exclusive_query_uses_uncertainty_fusion(self):
        args = make_args()
        args.semantic_tree_query_mode = "gaussian_shared_exclusive"
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_gaussian_query_sample = False
        args.semantic_tree_gaussian_query_condition_shared_variance = False
        args.lambda_semantic_tree_query_classification_aux = 0.1
        branch = SemanticTreeTransformerBranch(
            8,
            args=args,
            num_classes=2,
        ).train()
        with torch.no_grad():
            branch.shared_query_logvar.fill_(-6.0)
            branch.exclusive_query_logvar_head.weight.zero_()
            branch.exclusive_query_logvar_head.bias.fill_(2.0)

        original = torch.randn(5, 8)
        support = torch.randn(5, 8)
        deny = torch.randn(5, 8)
        depth = torch.tensor([0, 1, 2, 0, 1])
        batch = torch.tensor([0, 0, 0, 1, 1])
        graph, nodes = branch(
            original,
            support,
            deny,
            depth,
            batch,
            target=torch.tensor([1, 0]),
        )

        self.assertEqual(tuple(graph.shape), (2, 8))
        self.assertEqual(tuple(nodes.shape), (5, 8))
        self.assertEqual(
            tuple(branch.last_shared_attention.shape),
            (2, 1, 3),
        )
        self.assertEqual(
            tuple(branch.last_exclusive_attention.shape),
            (2, 1, 3),
        )
        self.assertEqual(
            tuple(branch.last_query_fusion_weights.shape),
            (2, 2),
        )
        self.assertTrue(
            torch.allclose(
                branch.last_query_fusion_weights.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.all(
                branch.last_query_fusion_weights[:, 0]
                > branch.last_query_fusion_weights[:, 1]
            )
        )
        self.assertEqual(
            tuple(branch.last_exclusive_query_mean.shape),
            (2, 1, 8),
        )
        self.assertTrue(torch.isfinite(branch.last_query_kl_loss))
        self.assertTrue(torch.isfinite(branch.last_query_shared_mi_loss))
        self.assertTrue(torch.isfinite(branch.last_query_exclusive_mi_loss))
        self.assertTrue(torch.isfinite(branch.last_query_diversity_loss))
        self.assertGreater(
            float(branch.last_query_classification_loss),
            0.0,
        )

    def test_gaussian_shared_exclusive_queries_are_graph_conditioned(
        self,
    ):
        args = make_args()
        args.semantic_tree_query_mode = "gaussian_shared_exclusive"
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_gaussian_query_sample = False
        args.semantic_tree_gaussian_query_shared_mi_temperature = 0.2
        args.semantic_tree_gaussian_query_exclusive_mi_temperature = 0.2
        args.lambda_semantic_tree_query_classification_aux = 0.1
        branch = SemanticTreeTransformerBranch(
            8,
            args=args,
            num_classes=2,
        ).train()
        with torch.no_grad():
            branch.shared_query_mean_head.weight.normal_(0.0, 0.1)
            branch.shared_query_logvar_head.weight.normal_(0.0, 0.05)

        original = torch.randn(8, 8)
        support = torch.randn(8, 8)
        deny = torch.randn(8, 8)
        depth = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        batch = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        target = torch.tensor([0, 0, 1, 1])
        branch(
            original,
            support,
            deny,
            depth,
            batch,
            target=target,
        )

        shared_mean = branch.last_shared_query_mean.flatten(start_dim=1)
        exclusive_mean = branch.last_exclusive_query_mean.flatten(
            start_dim=1
        )
        self.assertGreater(float(shared_mean.var(dim=0).sum()), 0.0)
        self.assertGreater(float(exclusive_mean.var(dim=0).sum()), 0.0)
        self.assertGreater(float(branch.last_query_shared_mi_loss), 0.0)
        self.assertGreater(float(branch.last_query_exclusive_mi_loss), 0.0)

        cross_graph_loss = (
            branch.last_query_shared_mi_loss
            + branch.last_query_exclusive_mi_loss
        )
        cross_graph_loss.backward()
        self.assertIsNotNone(branch.shared_query_mean_head.weight.grad)
        self.assertIsNotNone(branch.exclusive_query_mean_head.weight.grad)

    def test_learned_shared_gaussian_exclusive_uses_private_uncertainty(
        self,
    ):
        args = make_args()
        args.semantic_tree_query_mode = (
            "learned_shared_gaussian_exclusive"
        )
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_gaussian_query_sample = False
        args.lambda_semantic_tree_query_classification_aux = 0.1
        branch = SemanticTreeTransformerBranch(
            8,
            args=args,
            num_classes=2,
        ).train()
        with torch.no_grad():
            branch.exclusive_query_logvar_head.weight.zero_()
            branch.exclusive_query_logvar_head.bias.fill_(2.0)

        original = torch.randn(5, 8)
        support = torch.randn(5, 8)
        deny = torch.randn(5, 8)
        depth = torch.tensor([0, 1, 2, 0, 1])
        batch = torch.tensor([0, 0, 0, 1, 1])
        graph, nodes = branch(
            original,
            support,
            deny,
            depth,
            batch,
            target=torch.tensor([1, 0]),
        )

        self.assertEqual(tuple(graph.shape), (2, 8))
        self.assertEqual(tuple(nodes.shape), (5, 8))
        self.assertTrue(
            torch.allclose(
                branch.last_shared_query_mean[0],
                branch.last_shared_query_mean[1],
            )
        )
        self.assertIsNone(branch.last_shared_query_logvar)
        self.assertEqual(
            tuple(branch.last_exclusive_query_logvar.shape),
            (2, 1, 8),
        )
        self.assertTrue(
            torch.all(
                branch.last_query_fusion_weights[:, 0]
                > branch.last_query_fusion_weights[:, 1]
            )
        )
        self.assertGreater(float(branch.last_query_kl_loss), 0.0)
        self.assertIsNone(branch.last_query_shared_mi_loss)
        self.assertTrue(
            torch.isfinite(branch.last_query_exclusive_mi_loss)
        )
        self.assertGreater(
            float(branch.last_query_classification_loss),
            0.0,
        )

    def test_learned_shared_gaussian_exclusive_auxiliary_losses_backpropagate(
        self,
    ):
        args = make_args()
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = True
        args.classification_fusion_mode = "semantic_tree"
        args.semantic_tree_query_mode = (
            "learned_shared_gaussian_exclusive"
        )
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_gaussian_query_sample = True
        args.lambda_semantic_tree_query_kl_aux = 0.01
        args.lambda_semantic_tree_query_shared_mi_aux = 0.5
        args.lambda_semantic_tree_query_exclusive_mi_aux = 0.01
        args.lambda_semantic_tree_query_diversity_aux = 0.01
        args.lambda_semantic_tree_query_classification_aux = 0.01
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        branch = model.semantic_tree_transformer
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIsNotNone(branch.learned_query.grad)
        self.assertIsNone(branch.shared_query_logvar)
        self.assertIsNone(branch.shared_query_encoder)
        self.assertIsNotNone(branch.exclusive_query_mean_head.weight.grad)
        self.assertIsNotNone(branch.exclusive_query_logvar_head.weight.grad)
        self.assertGreater(
            float(model._last_semantic_tree_query_kl_loss),
            0.0,
        )
        self.assertEqual(
            float(model._last_semantic_tree_query_shared_mi_loss),
            0.0,
        )
        self.assertGreaterEqual(
            float(model._last_semantic_tree_query_exclusive_mi_loss),
            0.0,
        )
        self.assertGreater(
            float(model._last_semantic_tree_query_classification_loss),
            0.0,
        )

    def test_gaussian_shared_exclusive_query_auxiliary_losses_backpropagate(
        self,
    ):
        args = make_args()
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = True
        args.classification_fusion_mode = "semantic_tree"
        args.semantic_tree_query_mode = "gaussian_shared_exclusive"
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_gaussian_query_sample = True
        args.lambda_semantic_tree_query_kl_aux = 0.01
        args.lambda_semantic_tree_query_shared_mi_aux = 0.01
        args.lambda_semantic_tree_query_exclusive_mi_aux = 0.01
        args.lambda_semantic_tree_query_diversity_aux = 0.01
        args.lambda_semantic_tree_query_classification_aux = 0.01
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        branch = model.semantic_tree_transformer
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIsNotNone(branch.learned_query.grad)
        self.assertIsNotNone(branch.shared_query_logvar.grad)
        self.assertIsNotNone(branch.shared_query_mean_head.weight.grad)
        self.assertIsNotNone(branch.exclusive_query_mean_head.weight.grad)
        self.assertIsNotNone(branch.exclusive_query_logvar_head.weight.grad)
        self.assertGreater(
            float(model._last_semantic_tree_query_kl_loss),
            0.0,
        )
        self.assertGreaterEqual(
            float(model._last_semantic_tree_query_shared_mi_loss),
            0.0,
        )
        self.assertGreaterEqual(
            float(model._last_semantic_tree_query_exclusive_mi_loss),
            0.0,
        )
        self.assertGreaterEqual(
            float(model._last_semantic_tree_query_diversity_loss),
            0.0,
        )
        self.assertGreater(
            float(model._last_semantic_tree_query_classification_loss),
            0.0,
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_query_fusion_weights.shape),
            (2, 2),
        )

    def test_semantic_tree_original_keys_and_configurable_semantic_values(self):
        expected_input_dims = {
            "support_deny": 20,
            "support_deny_original": 28,
            "difference": 12,
            "support_deny_difference": 28,
            "original": 12,
        }
        for input_mode, expected_dim in expected_input_dims.items():
            args = make_args()
            args.semantic_tree_depth_dim = 4
            args.semantic_tree_input_mode = input_mode
            branch = SemanticTreeTransformerBranch(8, args=args)
            self.assertEqual(branch.key_projection[1].in_features, 8)
            self.assertEqual(
                branch.value_projection[1].in_features,
                expected_dim,
            )

    def test_semantic_tree_difference_value_modes_use_full_signed_change(self):
        original = torch.randn(3, 8)
        support = torch.randn(3, 8)
        deny = torch.randn(3, 8)
        depth = torch.randn(3, 4)
        difference = deny - support

        args = make_args()
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_input_mode = "difference"
        difference_branch = SemanticTreeTransformerBranch(8, args=args)
        difference_input = difference_branch._value_input(
            original,
            support,
            deny,
            depth,
        )
        self.assertTrue(
            torch.allclose(
                difference_input,
                torch.cat((difference, depth), dim=-1),
            )
        )

        args.semantic_tree_input_mode = "support_deny_difference"
        concatenated_branch = SemanticTreeTransformerBranch(8, args=args)
        concatenated_input = concatenated_branch._value_input(
            original,
            support,
            deny,
            depth,
        )
        self.assertTrue(
            torch.allclose(
                concatenated_input,
                torch.cat((support, deny, difference, depth), dim=-1),
            )
        )

    def test_semantic_tree_keys_ignore_dual_views_but_values_use_them(self):
        args = make_args()
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_input_mode = "support_deny"
        branch = SemanticTreeTransformerBranch(8, args=args).eval()
        original = torch.randn(3, 8)
        support = torch.randn(3, 8)
        deny = torch.randn(3, 8)
        depth = torch.tensor([0, 1, 1])
        batch = torch.zeros(3, dtype=torch.long)

        branch(original, support, deny, depth, batch)
        first_key = branch.last_key.clone()
        first_value = branch.last_value.clone()
        branch(original, support + 1.0, deny - 1.0, depth, batch)

        self.assertTrue(torch.allclose(first_key, branch.last_key))
        self.assertFalse(torch.allclose(first_value, branch.last_value))

    def test_semantic_tree_attention_penalizes_uncertain_change_keys(self):
        args = make_args()
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_bias_scale = 4.0
        branch = SemanticTreeTransformerBranch(8, args=args).eval()
        original = torch.zeros(3, 8)
        support = torch.randn(3, 8)
        deny = torch.randn(3, 8)
        batch = torch.zeros(3, dtype=torch.long)

        branch(
            original,
            support,
            deny,
            torch.tensor([0, 1, 1]),
            batch,
            change_node_uncertainty=torch.tensor([0.0, 10.0, 0.0]),
        )

        self.assertEqual(tuple(branch.last_uncertainty_bias.shape), (1, 1, 3))
        self.assertLess(
            float(branch.last_attention[0, 0, 1]),
            float(branch.last_attention[0, 0, 0]),
        )
        self.assertTrue(
            torch.all(branch.last_uncertainty_bias[:, :, 1] < 0.0)
        )
        self.assertTrue(
            torch.allclose(
                branch.last_uncertainty_bias[:, :, 0],
                torch.zeros(1, 1),
            )
        )

    def test_zero_semantic_tree_uncertainty_scale_disables_bias(self):
        args = make_args()
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_source = "edge_relation"
        args.semantic_tree_uncertainty_bias_scale = 0.0
        branch = SemanticTreeTransformerBranch(8, args=args).eval()
        original = torch.randn(3, 8)
        support = torch.randn(3, 8)
        deny = torch.randn(3, 8)
        batch = torch.zeros(3, dtype=torch.long)

        branch(
            original,
            support,
            deny,
            torch.tensor([0, 1, 1]),
            batch,
            change_node_uncertainty=torch.tensor([0.0, 1.0, 0.5]),
        )

        self.assertTrue(branch.uncertainty_bias_active)
        self.assertTrue(
            torch.allclose(
                branch.last_uncertainty_bias,
                torch.zeros(1, 1, 3),
            )
        )

    def test_semantic_tree_change_mi_and_gaussian_uncertainty_are_used(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_gaussian_semantic_change_bottleneck = True
        args.lambda_semantic_tree_change_mi_aux = 0.05
        args.use_semantic_tree_change_uncertainty_bias = True
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()

        output, _, _, _ = model(make_batch())
        loss = model._last_aux_loss
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertGreater(
            float(model._last_semantic_tree_change_mi_loss),
            0.0,
        )
        self.assertEqual(
            tuple(model._last_change_node_uncertainty.shape),
            (5,),
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_uncertainty_bias.shape),
            (2, 1, 3),
        )
        self.assertIsNotNone(
            model.semantic_tree_transformer.key_projection[1].weight.grad
        )

    def test_semantic_tree_can_reuse_pre_route_edge_uncertainty(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_uncertainty_sampling = False
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_gaussian_semantic_change_bottleneck = True
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_source = "edge_relation"
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
        expected = torch.stack(
            (
                model._last_edge_uncertainty.new_zeros(()),
                model._last_edge_uncertainty[0],
                model._last_edge_uncertainty[1],
                model._last_edge_uncertainty.new_zeros(()),
                model._last_edge_uncertainty[2],
            )
        )

        self.assertTrue(
            torch.allclose(
                model._last_semantic_tree_node_uncertainty,
                expected,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_node_uncertainty,
                expected,
                atol=1e-6,
            )
        )
        self.assertEqual(
            model.semantic_tree_transformer.uncertainty_source,
            "edge_relation",
        )
        self.assertIsNotNone(model._last_semantic_tree_uncertainty_bias)
        self.assertTrue(torch.allclose(model._last_keep_sample, torch.ones(3)))

    def test_reliability_hinge_requires_semantic_tree_uncertainty_bias(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_semantic_tree_reliability_hinge_loss = True
        args.use_semantic_tree_change_uncertainty_bias = False

        with self.assertRaisesRegex(
            ValueError,
            "requires use_semantic_tree_change_uncertainty_bias",
        ):
            BiGCN_UncertaintySemanticChange(
                in_feats=5,
                hid_feats=8,
                out_feats=8,
                num_classes=2,
                args=args,
                device=torch.device("cpu"),
            )

    def test_restricted_reliability_hinge_enters_auxiliary_loss(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_uncertainty_sampling = False
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_source = "edge_relation"
        args.semantic_tree_change_uncertainty_detach = True
        args.use_semantic_tree_reliability_hinge_loss = True
        # A large margin deterministically activates every graph that contains
        # both reliable and uncertain evidence in this smoke test.
        args.semantic_tree_reliability_hinge_margin = 10.0
        args.lambda_semantic_tree_reliability_hinge_aux = 0.2
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()

        output, _, _, _ = model(make_batch())
        hinge_loss = model._last_semantic_tree_reliability_hinge_loss
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(hinge_loss))
        self.assertGreater(float(hinge_loss), 0.0)
        self.assertAlmostEqual(
            float(model._last_semantic_tree_reliability_hinge_active_rate),
            1.0,
            places=6,
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_reliable_evidence.shape),
            (2, 8),
        )
        self.assertEqual(
            tuple(model._last_semantic_tree_uncertain_evidence.shape),
            (2, 8),
        )

        model.auxiliary_loss().backward()
        self.assertIsNotNone(
            model.semantic_tree_reliability_classifier.weight.grad
        )
        self.assertIsNotNone(
            model.semantic_tree_transformer.value_projection[1].weight.grad
        )

    def test_restricted_reliability_hinge_stops_after_margin_is_met(self):
        args = make_args()
        args.use_trend_graph = False
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_source = "edge_relation"
        args.use_semantic_tree_reliability_hinge_loss = True
        args.semantic_tree_reliability_hinge_margin = 0.1
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()

        branch = model.semantic_tree_transformer
        branch.graph_output = torch.nn.Identity()
        branch.last_attention_probability = torch.tensor(
            [[[0.5, 0.5]]],
            dtype=torch.float32,
        )
        branch.last_value = torch.zeros(1, 2, 8)
        branch.last_value[0, 0, 0] = 10.0
        branch.last_value[0, 1, 1] = 10.0
        branch.last_valid_mask = torch.ones(1, 2, dtype=torch.bool)
        branch.last_change_uncertainty = torch.tensor([[0.0, 1.0e6]])
        with torch.no_grad():
            classifier = model.semantic_tree_reliability_classifier
            classifier.weight.zero_()
            classifier.bias.zero_()
            classifier.weight[0, 0] = 1.0
            classifier.weight[1, 1] = 1.0

        outputs = model._semantic_tree_reliability_hinge_loss(
            torch.tensor([0])
        )
        self.assertLess(
            float(
                outputs["reliable_ce"]
                + args.semantic_tree_reliability_hinge_margin
            ),
            float(outputs["uncertain_ce"]),
        )
        self.assertEqual(float(outputs["active_rate"]), 0.0)
        self.assertEqual(float(outputs["raw_loss"]), 0.0)
        self.assertEqual(float(outputs["loss"]), 0.0)

    def test_change_pool_and_semantic_tree_share_edge_uncertainty(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_uncertainty_sampling = False
        args.use_node_keep_in_change_pool = False
        args.use_change_uncertainty_pooling = True
        args.change_uncertainty_pool_scale = 1.0
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_source = "edge_relation"
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
        shared_uncertainty = model._last_node_uncertainty
        expected_reliability = torch.exp(
            -shared_uncertainty / (1.0 + shared_uncertainty)
        )
        expected_change_graph = model._pool_root_connected_nodes(
            model._last_change_nodes,
            expected_reliability,
            data.batch,
        )

        self.assertTrue(
            torch.allclose(
                model._last_semantic_tree_node_uncertainty,
                shared_uncertainty,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_change_pool_reliability,
                expected_reliability,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_change_graph,
                expected_change_graph,
                atol=1e-6,
            )
        )

    def test_forward_does_not_gate_parity_views_with_parent_edge_twice(self):
        args = make_args()
        args.use_trend_graph = False
        args.use_uncertainty_sampling = False
        args.use_semantic_tree_transformer = True
        args.semantic_tree_transformer_heads = 2
        args.semantic_tree_transformer_layers = 1
        args.semantic_tree_depth_dim = 4
        args.classification_fusion_mode = "change_semantic_tree"
        args.use_semantic_tree_change_uncertainty_bias = True
        args.semantic_tree_uncertainty_source = "edge_relation"
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("post-parity semantic node gate was called")

        model._build_semantic_node_weights = fail_if_called
        output, _, _, _ = model(make_batch())

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIsNotNone(model._last_semantic_tree_uncertainty_bias)
        self.assertGreater(
            float(model._last_semantic_tree_uncertainty_bias.abs().sum()),
            0.0,
        )

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
        self.assertEqual(tuple(model._last_original_nodes.shape), (5, 8))
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

    def test_parity_view_pooling_composes_root_paths(self):
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
                    x=torch.randn(4, 5),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                    edge_stance=torch.tensor([0, 0, 1]),
                    y=torch.tensor([0]),
                    num_hop=torch.tensor([3]),
                    user_state=torch.zeros(1, 4, 3),
                ),
                Data(
                    x=torch.randn(4, 5),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                    edge_stance=torch.tensor([1, 0, 1]),
                    y=torch.tensor([0]),
                    num_hop=torch.tensor([3]),
                    user_state=torch.zeros(1, 4, 3),
                ),
            ]
        )
        support_weight = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        deny_weight = 1.0 - support_weight

        support_node_weight, deny_node_weight = (
            model._build_parity_view_node_weights(
                data,
                support_weight,
                deny_weight,
            )
        )

        self.assertTrue(
            torch.equal(
                support_node_weight,
                torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                deny_node_weight,
                torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            )
        )

    def test_semantic_node_weight_mode_defaults_to_local_edges(self):
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
                    y=torch.tensor([0]),
                    num_hop=torch.tensor([2]),
                    user_state=torch.zeros(1, 4, 3),
                )
            ]
        )
        support_weight = torch.tensor([0.0, 0.0])
        deny_weight = torch.tensor([1.0, 1.0])

        support_node_weight, deny_node_weight = (
            model._build_semantic_node_weights(
                data,
                support_weight,
                deny_weight,
            )
        )

        self.assertEqual(model.semantic_node_weight_mode, "local")
        self.assertTrue(
            torch.equal(
                support_node_weight,
                torch.tensor([1.0, 0.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                deny_node_weight,
                torch.tensor([1.0, 1.0, 1.0]),
            )
        )

        args.semantic_node_weight_mode = "root_parity"
        parity_model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        support_node_weight, deny_node_weight = (
            parity_model._build_semantic_node_weights(
                data,
                support_weight,
                deny_weight,
            )
        )
        self.assertTrue(
            torch.equal(
                support_node_weight,
                torch.tensor([1.0, 0.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                deny_node_weight,
                torch.tensor([1.0, 1.0, 0.0]),
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

    def test_dirichlet_relation_routing_forward_records_alpha(self):
        args = make_args()
        args.use_dirichlet_relation_routing = True
        args.use_uncertainty_sampling = False
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

        self.assertEqual(tuple(model._last_edge_dirichlet_alpha.shape), (3, 2))
        self.assertTrue((model._last_edge_dirichlet_alpha > 0).all())
        self.assertIsNone(model._last_edge_masses)
        self.assertIsNone(model._last_edge_unknown_mass)
        self.assertTrue(
            torch.allclose(
                model._last_edge_probabilities.sum(dim=-1),
                torch.ones(3),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())

    def test_gaussian_semantic_change_bottleneck_adds_auxiliary_kl(self):
        args = make_args()
        args.use_gaussian_semantic_change_bottleneck = True
        args.semantic_change_gaussian_sample = False
        args.lambda_semantic_change_bottleneck = 0.01
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

        self.assertIsNotNone(
            model.semantic_change_encoder.mean_head.weight.grad
        )
        self.assertIsNotNone(model._last_semantic_change_bottleneck_loss)
        self.assertGreaterEqual(
            float(model._last_semantic_change_bottleneck_loss),
            0.0,
        )
        self.assertEqual(tuple(model._last_change_nodes.shape), (5, 8))
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())


class ResGCNUncertaintySemanticChangeTest(unittest.TestCase):
    def test_resgcn_supports_probabilistic_sgcn_semantic_views(self):
        args = make_args()
        args.semantic_parity_aggregation = "probabilistic_sgcn"
        args.use_uncertainty_sampling = False
        model = ResGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()
        model.set_epoch(model.edge_router.warmup_epochs)

        output, unknown, support, deny = model(make_batch())

        self.assertIsInstance(
            model.semantic_parity_encoder.top_down,
            SemanticParityProbabilisticSGCNDirectionEncoder,
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(unknown).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(torch.isfinite(deny).all())

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

    def test_resgcn_supports_learned_shared_gaussian_exclusive_query(
        self,
    ):
        args = make_args()
        args.use_trend_graph = False
        args.use_semantic_tree_transformer = True
        args.classification_fusion_mode = "change_semantic_tree"
        args.semantic_tree_query_mode = (
            "learned_shared_gaussian_exclusive"
        )
        args.semantic_tree_depth_dim = 4
        args.semantic_tree_gaussian_query_sample = False
        model = ResGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        branch = model.semantic_tree_transformer
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIsNone(branch.last_shared_query_logvar)
        self.assertEqual(
            tuple(branch.last_exclusive_query_logvar.shape),
            (2, 1, 8),
        )
        self.assertEqual(
            tuple(branch.last_query_fusion_weights.shape),
            (2, 2),
        )


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
