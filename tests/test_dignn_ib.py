"""Numerical bounds, gradient routing, inference and v1 ablation for WL VIB."""

import ast
import io
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from model.DIGNN import DIGNN, TopologyInformationBottleneck


def model(**options):
    settings = dict(dropout=0.0, max_hop=3, dignn_wl_clusters=[6, 3],
                    dignn_wl_buckets=128, dignn_ib_enabled=True,
                    dignn_ib_beta=0.001, dignn_ib_classification_weight=0.1,
                    dignn_ib_warmup_epochs=10)
    settings.update(options)
    return DIGNN(4, 8, 2, SimpleNamespace(**settings))


def events():
    graphs = []
    for n, label in ((4, 0), (3, 1), (1, 0)):
        edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
        graphs.append(Data(x=torch.randn(n, 4), edge_index=edges,
                           directed_edge_index=edges.clone(),
                           directed_edge_stance=torch.arange(n - 1) % 2,
                           y=torch.tensor([label]), root_index=torch.tensor([0])))
    return graphs


class TopologyIBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(29)

    def test_kl_matches_distribution_reference_and_reduction(self):
        mean = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        log_var = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        posterior = torch.distributions.Normal(mean, (0.5 * log_var).exp())
        prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        expected = torch.distributions.kl_divergence(posterior, prior).sum(-1).mean()
        actual = TopologyInformationBottleneck.prior_kl(mean, log_var)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(mean.grad).all()))
        self.assertTrue(bool(torch.isfinite(log_var.grad).all()))
        self.assertEqual(float(TopologyInformationBottleneck.prior_kl(torch.zeros(2, 8),
                                                                    torch.zeros(2, 8))), 0)
        self.assertEqual(float(TopologyInformationBottleneck.prior_kl(torch.ones(2, 8),
                                                                    torch.zeros(2, 8))), 4)

    def test_sampling_in_training_mean_in_eval_and_variance_bounds(self):
        bottleneck = TopologyInformationBottleneck(8)
        hidden = torch.randn(3, 8)
        first, second = bottleneck(hidden), bottleneck(hidden)
        self.assertFalse(torch.equal(first['sample'], second['sample']))
        torch.testing.assert_close(first['mean'], hidden)
        bottleneck.eval()
        torch.testing.assert_close(bottleneck(hidden)['sample'], hidden)
        for bias, expected in ((1000., 6.), (-1000., -10.)):
            with torch.no_grad():
                bottleneck.log_variance.bias.fill_(bias)
            result = bottleneck(hidden)
            torch.testing.assert_close(result['log_variance'], torch.full_like(hidden, expected))
            self.assertTrue(bool(torch.isfinite(result['kl'])))

    def test_total_loss_coefficients_and_epoch_ramp(self):
        network = model().train()
        batch = Batch.from_data_list(events())
        for epoch, expected_beta in ((1, 0.0001), (5, 0.0005), (10, 0.001), (40, 0.001)):
            network.set_epoch(epoch)
            output = network(batch)[0]
            diagnostics = network.get_diagnostics()
            self.assertAlmostEqual(network.loss_weights['ib_kl'], expected_beta)
            expected = (0.1 * diagnostics['ib_classification'] + expected_beta * diagnostics['ib_kl'])
            torch.testing.assert_close(network.auxiliary_loss().detach(), expected)
            torch.testing.assert_close(network.classification_loss(output, batch.y), F.nll_loss(output, batch.y))
        no_ramp = model(dignn_ib_warmup_epochs=0)
        self.assertEqual(no_ramp.loss_weights['ib_kl'], 0.001)

    def test_training_updates_bottleneck_encoder_aux_head_and_fusion(self):
        network = model().train()
        optimizer = network.init_optimizer()
        batch = Batch.from_data_list(events())
        before = network.topology_bottleneck.log_variance.weight.detach().clone()
        loss = network.compute_loss(batch)
        loss.backward()
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for module in (network.topology_encoder, network.topology_bottleneck.mean,
                       network.topology_bottleneck.log_variance, network.topology_classifier,
                       network.text_encoder, network.view_attention):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, network.topology_bottleneck.log_variance.weight))

    def test_auxiliary_terms_train_topology_without_fusion_and_do_not_train_text(self):
        # Individually exercise compression and relevance, without fused CE.
        for beta, weight in ((0.001, 0.), (0., 0.1)):
            network = model(dignn_ib_beta=beta, dignn_ib_classification_weight=weight).train()
            network(Batch.from_data_list(events()))
            network.auxiliary_loss().backward()
            for module in (network.topology_encoder, network.topology_bottleneck.mean,
                           network.topology_bottleneck.log_variance):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()
                                       if p.grad is not None), 0)
            for parameter in network.text_encoder.parameters():
                self.assertTrue(parameter.grad is None or float(parameter.grad.abs().sum()) == 0)

    def test_targets_affect_only_aux_loss_and_training_requires_targets(self):
        network = model(class_weights=[1., 2.]).train()
        batch = Batch.from_data_list(events())
        torch.manual_seed(55)
        first = network(batch)[0]
        torch.manual_seed(55)
        views = network.encode_views(batch)
        expected_ce = F.cross_entropy(views['ib_logits'], batch.y, weight=torch.tensor([1., 2.]))
        torch.testing.assert_close(network.get_diagnostics()['ib_classification'], expected_ce.detach())
        changed = batch.clone()
        changed.y = 1 - changed.y
        torch.manual_seed(55)
        torch.testing.assert_close(first, network(changed)[0])
        del changed.y
        with self.assertRaisesRegex(ValueError, 'IB training requires'):
            network(changed)
        network.eval()
        self.assertTrue(bool(torch.isfinite(network(changed)[0]).all()))
        self.assertEqual(float(network.auxiliary_loss()), 0)
        self.assertNotIn('ib_classification', network.get_diagnostics())

    def test_evaluation_is_deterministic_batch_invariant_and_checkpointable(self):
        network = model().eval()
        graphs = events()
        batch = Batch.from_data_list(graphs)
        expected = network(batch)[0]
        torch.testing.assert_close(network(batch)[0], expected)
        torch.testing.assert_close(torch.cat([network(g)[0] for g in graphs]), expected)
        views = network.encode_views(batch)
        torch.testing.assert_close(views['topology_graph'], views['ib_mean'])
        stream = io.BytesIO()
        torch.save(network.state_dict(), stream)
        stream.seek(0)
        restored = model().eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(batch)[0], expected)

    def test_disabled_ib_is_exact_v1_and_zero_beta_is_not_v1(self):
        settings = dict(dropout=0.0, max_hop=3, dignn_wl_clusters=[6, 3], dignn_wl_buckets=128)
        torch.manual_seed(12)
        original = DIGNN(4, 8, 2, SimpleNamespace(**settings)).eval()
        torch.manual_seed(12)
        ablation = model(dignn_ib_enabled=False).eval()
        self.assertEqual(set(original.state_dict()), set(ablation.state_dict()))
        ablation.load_state_dict(original.state_dict(), strict=True)
        batch = Batch.from_data_list(events())
        torch.testing.assert_close(original(batch)[0], ablation(batch)[0])
        self.assertIsNone(ablation.topology_bottleneck)
        self.assertIsNotNone(model(dignn_ib_beta=0).topology_bottleneck)

    def test_invalid_configuration_is_rejected(self):
        for settings in ({'dignn_ib_beta': -1}, {'dignn_ib_beta': float('nan')},
                         {'dignn_ib_classification_weight': float('inf')},
                         {'dignn_ib_warmup_epochs': -1}, {'dignn_ib_warmup_epochs': 1.5},
                         {'dignn_ib_enabled': 'false'}, {'dignn_topology_features': 'energy'},
                         {'class_weights': [1.]}, {'class_weights': [1., -1.]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                model(**settings)

    def test_trainer_reports_the_actual_ib_loss_terms(self):
        path = Path(__file__).resolve().parents[1] / 'trainer' / 'EIN_trainer.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        trainer_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                             and node.name == 'EINTrainer')
        names = {'_tensor_mean', '_model_diagnostics'}
        methods = [node for node in trainer_class.body if isinstance(node, ast.FunctionDef)
                   and node.name in names]
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        network = model().train()
        network(Batch.from_data_list(events()))
        trainer = SimpleNamespace(model=network)
        trainer._tensor_mean = MethodType(namespace['_tensor_mean'], trainer)
        diagnostics = namespace['_model_diagnostics'](trainer)
        for name in ('ib_kl', 'ib_classification', 'ib_kl_weight', 'ib_posterior_std'):
            self.assertAlmostEqual(diagnostics[name], float(network.get_diagnostics()[name]))


if __name__ == '__main__':
    unittest.main()
