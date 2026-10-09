"""Behavioral checks for separated inputs, graph batches and training losses."""

import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import batched_negative_sampling

from model.DIGNN import DIGNN, normalized_hsic


def graph(nodes, edges, label=0):
    edges = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
    return Data(x=torch.randn(nodes, 4), edge_index=edges,
                y=torch.tensor([label]), root_index=torch.tensor([0]))


def make_model(**kwargs):
    settings = dict(n_layers_conv=2, dropout=0.0, max_hop=3)
    settings.update(kwargs)
    return DIGNN(4, 8, 2, SimpleNamespace(**settings))


class DIGNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        random.seed(17)

    def test_topology_ignores_text_and_text_ignores_propagation_edges(self):
        model = make_model().eval()
        data = graph(4, [[0, 0, 0], [1, 2, 3]])
        first = model.encode_views(data)
        changed_text = data.clone()
        changed_text.x *= -10
        second = model.encode_views(changed_text)
        torch.testing.assert_close(first["topology_graph"], second["topology_graph"])
        self.assertFalse(torch.allclose(first["text_graph"], second["text_graph"]))
        chain = data.clone()
        chain.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        third = model.encode_views(chain)
        torch.testing.assert_close(first["text_graph"], third["text_graph"])
        self.assertFalse(torch.allclose(first["topology_graph"], third["topology_graph"]))

    def test_input_gradients_do_not_cross_view_encoders(self):
        model = make_model().eval()
        data = graph(3, [[0, 0], [1, 2]])
        data.x.requires_grad_(True)
        views = model.encode_views(data)
        self.assertIsNone(torch.autograd.grad(views["topology_graph"].square().sum(),
                                              data.x, allow_unused=True)[0])
        grad = torch.autograd.grad(views["text_graph"][0, 0], data.x)[0]
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_batch_predictions_match_single_events_including_isolates(self):
        model = make_model().eval()
        graphs = [graph(4, [[0, 1, 1], [1, 2, 3]]), graph(1, [[], []]),
                  graph(3, [[0], [1]])]
        batch = Batch.from_data_list(graphs)
        output, u, s, d = model(batch)
        single = torch.cat([model(event)[0] for event in graphs])
        torch.testing.assert_close(output, single)
        torch.testing.assert_close(output.exp().sum(-1), torch.ones(3))
        self.assertEqual(u.shape, (3, 3, 1))
        self.assertEqual(s.shape, u.shape)
        self.assertEqual(d.shape, u.shape)

    def test_node_permutation_with_preserved_root_does_not_change_predictions(self):
        model = make_model().eval()
        original = graph(5, [[0, 0, 1, 3], [1, 2, 3, 4]])
        order = torch.tensor([4, 2, 0, 3, 1])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(5)
        permuted = Data(x=original.x[order], edge_index=inverse[original.edge_index],
                        root_index=inverse[original.root_index])
        torch.testing.assert_close(model(original)[0], model(permuted)[0])

    def test_directed_cache_edges_take_priority_and_duplicates_are_coalesced(self):
        model = make_model().eval()
        data = graph(3, [[0, 1], [1, 2]])
        data.directed_edge_index = data.edge_index.clone()
        expected = model(data)[0]
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.testing.assert_close(expected, model(data)[0])
        data.directed_edge_index = torch.cat((data.directed_edge_index,
                                             data.directed_edge_index,
                                             torch.tensor([[0, 1, 2], [0, 1, 2]])), dim=1)
        torch.testing.assert_close(expected, model(data)[0])

    def test_inference_needs_no_labels_stances_or_states_and_clears_losses(self):
        model = make_model().train()
        data = graph(3, [[0, 0], [1, 2]])
        model(data)
        self.assertGreater(float(model.auxiliary_loss().detach()), 0)
        model.eval()
        expected = model(data)[0]
        del data.y
        torch.testing.assert_close(expected, model(data)[0])
        self.assertEqual(float(model.auxiliary_loss()), 0)
        self.assertEqual(set(model.get_diagnostics()), {"view_attention"})
        self.assertEqual(float(model.physics_loss(None, None, None, None)), 0)

    def test_losses_send_finite_gradients_to_encoders_fusion_and_decoders(self):
        model = make_model(dropout=0.2).train()
        data = Batch.from_data_list([graph(4, [[0, 0, 1], [1, 2, 3]]),
                                     graph(3, [[0, 1], [1, 2]], 1), graph(1, [[], []])])
        optimizer = model.init_optimizer()
        before = model.classifier.weight.detach().clone()
        output = model(data)[0]
        loss = model.classification_loss(output, data.y) + model.auxiliary_loss()
        diagnostics = model.get_diagnostics()
        expected_aux = sum(model.loss_weights[name] * diagnostics[name] for name in model.loss_weights)
        torch.testing.assert_close(model.auxiliary_loss().detach(), expected_aux)
        torch.testing.assert_close(diagnostics["view_attention"].sum(-1), torch.ones(3))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for module in (model.topology_encoder, model.text_encoder, model.view_attention,
                       model.text_decoder, model.structure_decoder, model.edge_source, model.edge_target):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_negative_edges_are_local_unique_and_not_positive_or_self_loops(self):
        model = make_model().train()
        data = Batch.from_data_list([graph(3, [[0, 0], [1, 2]]), graph(1, [[], []]),
                                     graph(4, [[0, 1, 2], [1, 2, 3]])])
        sampled = []

        def capture(*args, **kwargs):
            negatives = batched_negative_sampling(*args, **kwargs)
            sampled.append(negatives)
            return negatives

        with patch("model.DIGNN.batched_negative_sampling", side_effect=capture):
            loss = model.compute_loss(data)
        self.assertTrue(bool(torch.isfinite(loss)))
        negative = sampled[0]
        self.assertGreater(negative.size(1), 0)
        self.assertTrue(torch.equal(data.batch[negative[0]], data.batch[negative[1]]))
        self.assertFalse(bool((negative[0] == negative[1]).any()))
        pos_pairs = set(map(tuple, data.edge_index.t().tolist()))
        neg_pairs = list(map(tuple, negative.t().tolist()))
        self.assertTrue(pos_pairs.isdisjoint(neg_pairs))
        self.assertEqual(len(neg_pairs), len(set(neg_pairs)))

    def test_empty_and_complete_events_have_finite_training_losses(self):
        for data in (graph(1, [[], []]), graph(3, [[], []]),
                     graph(2, [[0, 1], [1, 0]])):
            model = make_model().train()
            loss = model.compute_loss(data)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                                if p.grad is not None))

    def test_reconstruction_is_balanced_per_event_instead_of_per_node(self):
        prediction = torch.zeros(5, 4)
        target = torch.tensor([[1.] * 4] + [[3.] * 4] * 4)
        loss = DIGNN._event_mse(prediction, target, torch.tensor([0, 1, 1, 1, 1]), 2)
        self.assertEqual(float(loss), (1 + 9) / 2)

    def test_disabling_regularizers_avoids_negative_sampling(self):
        options = dict(dignn_text_recon_weight=0, dignn_structure_recon_weight=0,
                       dignn_edge_recon_weight=0, dignn_independence_weight=0)
        model = make_model(**options).train()
        with patch("model.DIGNN.batched_negative_sampling", side_effect=AssertionError("unexpected sampling")):
            model(graph(3, [[0, 0], [1, 2]]))
        self.assertEqual(float(model.auxiliary_loss().detach()), 0)

    def test_fixed_fusion_ablations_use_requested_view(self):
        data = graph(3, [[0, 0], [1, 2]])
        for mode, weights in (("mean", [0.5, 0.5]), ("topology", [1., 0.]), ("text", [0., 1.])):
            model = make_model(dignn_fusion=mode).eval()
            views = model.encode_views(data)
            representation = weights[0] * views["topology_graph"] + weights[1] * views["text_graph"]
            expected = F.log_softmax(model.classifier(representation), dim=-1)
            torch.testing.assert_close(expected, model(data)[0])

    def test_invalid_edges_and_roots_are_rejected(self):
        model = make_model()
        data = Batch.from_data_list([graph(2, [[0], [1]]), graph(2, [[0], [1]])])
        data.edge_index = torch.tensor([[0], [2]])
        with self.assertRaisesRegex(ValueError, "different events"):
            model(data)
        data.edge_index = torch.tensor([[0], [4]])
        with self.assertRaisesRegex(ValueError, "outside x"):
            model(data)
        data.edge_index = torch.empty(2, 0, dtype=torch.long)
        data.root_index = torch.tensor([0, 0])
        with self.assertRaisesRegex(ValueError, "globally indexed root"):
            model(data)


class HSICTest(unittest.TestCase):
    def test_constants_and_small_batches_return_zero_with_finite_gradients(self):
        for count in (1, 2, 5):
            a = torch.ones(count, 4, requires_grad=True)
            b = torch.randn(count, 4, requires_grad=True)
            loss = normalized_hsic(a, b)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(a.grad).all()))
            self.assertTrue(bool(torch.isfinite(b.grad).all()))

    def test_dependent_views_score_higher_than_independent_views(self):
        torch.manual_seed(11)
        a = torch.randn(64, 4, requires_grad=True)
        b = torch.randn(64, 4, requires_grad=True)
        same = normalized_hsic(a, a)
        different = normalized_hsic(a, b)
        self.assertAlmostEqual(float(same.detach()), 1.0, places=5)
        self.assertLess(float(different.detach()), 0.4)
        different.backward()
        self.assertGreater(float(a.grad.abs().sum()), 0)
        self.assertGreater(float(b.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(a.grad).all()))
        self.assertTrue(bool(torch.isfinite(b.grad).all()))


if __name__ == "__main__":
    unittest.main()
