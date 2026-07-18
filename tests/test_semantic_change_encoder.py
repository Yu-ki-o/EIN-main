import unittest

import torch

from model.semantic_change_encoder import (
    DPGASemanticChangeEncoder,
    GaussianSemanticChangeBottleneck,
    MLPSemanticChangeEncoder,
    build_semantic_change_encoder,
)


class MLPSemanticChangeEncoderTest(unittest.TestCase):
    def test_output_shape_and_gradient(self):
        encoder = MLPSemanticChangeEncoder(
            input_dim=4,
            output_dim=6,
            hidden_dim=8,
            dropout=0.0,
        )
        support = torch.randn(5, 4, requires_grad=True)
        deny = torch.randn(5, 4, requires_grad=True)

        change = encoder(support, deny)
        self.assertEqual(tuple(change.shape), (5, 6))

        change.sum().backward()
        self.assertIsNotNone(support.grad)
        self.assertIsNotNone(deny.grad)

    def test_identical_views_produce_zero_change(self):
        encoder = MLPSemanticChangeEncoder(input_dim=4).eval()
        nodes = torch.randn(7, 4)

        change = encoder(nodes, nodes)
        self.assertTrue(torch.equal(change, torch.zeros_like(change)))

    def test_features_keep_direction_and_magnitude(self):
        encoder = MLPSemanticChangeEncoder(input_dim=2)
        support = torch.tensor([[1.0, -2.0]])
        deny = torch.tensor([[3.0, -5.0]])

        forward_features = encoder.change_features(support, deny)
        reverse_features = encoder.change_features(deny, support)

        self.assertTrue(
            torch.equal(forward_features[:, :2], -reverse_features[:, :2])
        )
        self.assertTrue(
            torch.equal(forward_features[:, 2:], reverse_features[:, 2:])
        )

    def test_factory_exposes_replaceable_interface(self):
        encoder = build_semantic_change_encoder(
            "mlp",
            input_dim=4,
            output_dim=4,
        )
        self.assertIsInstance(encoder, MLPSemanticChangeEncoder)

    def test_mismatched_views_are_rejected(self):
        encoder = MLPSemanticChangeEncoder(input_dim=4)
        with self.assertRaises(ValueError):
            encoder(torch.randn(3, 4), torch.randn(4, 4))


class DPGASemanticChangeEncoderTest(unittest.TestCase):
    def test_output_shape_and_gradient_with_batch_context(self):
        encoder = DPGASemanticChangeEncoder(
            input_dim=4,
            output_dim=6,
            hidden_dim=8,
            dropout=0.0,
            pseudo_nodes=3,
            layers=1,
        )
        support = torch.randn(5, 4, requires_grad=True)
        deny = torch.randn(5, 4, requires_grad=True)
        batch = torch.tensor([0, 0, 1, 1, 1])
        support_weight = torch.tensor([1.0, 0.7, 1.0, 0.2, 0.4])
        deny_weight = torch.tensor([1.0, 0.3, 1.0, 0.8, 0.6])
        node_keep = torch.tensor([1.0, 0.5, 1.0, 1.0, 0.25])

        change = encoder(
            support,
            deny,
            batch=batch,
            support_node_weight=support_weight,
            deny_node_weight=deny_weight,
            node_keep=node_keep,
        )
        self.assertEqual(tuple(change.shape), (5, 6))

        change.sum().backward()
        self.assertIsNotNone(support.grad)
        self.assertIsNotNone(deny.grad)

    def test_identical_views_produce_zero_change(self):
        encoder = DPGASemanticChangeEncoder(
            input_dim=4,
            hidden_dim=8,
            dropout=0.0,
            pseudo_nodes=2,
            layers=1,
        ).eval()
        nodes = torch.randn(7, 4)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])

        change = encoder(nodes, nodes, batch=batch)
        self.assertTrue(torch.equal(change, torch.zeros_like(change)))

    def test_factory_exposes_dpga_option(self):
        encoder = build_semantic_change_encoder(
            "dpga",
            input_dim=4,
            output_dim=4,
            hidden_dim=8,
        )
        self.assertIsInstance(encoder, DPGASemanticChangeEncoder)


class GaussianSemanticChangeBottleneckTest(unittest.TestCase):
    def test_output_shape_gradient_and_kl(self):
        encoder = GaussianSemanticChangeBottleneck(
            input_dim=4,
            output_dim=6,
            hidden_dim=8,
            latent_dim=5,
            dropout=0.0,
        )
        support = torch.randn(5, 4, requires_grad=True)
        deny = torch.randn(5, 4, requires_grad=True)

        change = encoder(support, deny)
        loss = change.sum() + encoder.kl_loss()
        loss.backward()

        self.assertEqual(tuple(change.shape), (5, 6))
        self.assertEqual(tuple(encoder.last_mean.shape), (5, 5))
        self.assertEqual(tuple(encoder.last_logvar.shape), (5, 5))
        self.assertGreaterEqual(float(encoder.kl_loss()), 0.0)
        self.assertIsNotNone(support.grad)
        self.assertIsNotNone(deny.grad)

    def test_identical_views_produce_zero_change_without_sampling(self):
        encoder = GaussianSemanticChangeBottleneck(input_dim=4).eval()
        nodes = torch.randn(7, 4)

        change = encoder(nodes, nodes)

        self.assertTrue(torch.equal(change, torch.zeros_like(change)))
        self.assertAlmostEqual(float(encoder.kl_loss()), 0.0, places=6)

    def test_factory_switch_replaces_mlp_with_gaussian_bottleneck(self):
        class Args:
            use_gaussian_semantic_change_bottleneck = True
            semantic_change_gaussian_latent_dim = 3
            semantic_change_gaussian_sample = False

        encoder = build_semantic_change_encoder(
            "mlp",
            input_dim=4,
            output_dim=4,
            hidden_dim=8,
            args=Args(),
        )

        self.assertIsInstance(encoder, GaussianSemanticChangeBottleneck)
        self.assertEqual(encoder.latent_dim, 3)


if __name__ == "__main__":
    unittest.main()
