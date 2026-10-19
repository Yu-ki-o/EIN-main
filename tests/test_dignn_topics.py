"""Behavioral checks for source-conditioned soft topics and typed coarse graphs."""
import io
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Batch, Data

from model.DIGNN import DIGNN, SoftTopicEncoder


def graph(n=6, label=0):
    edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
    return Data(x=torch.randn(n, 4), edge_index=edges,
                directed_edge_index=edges.clone(),
                directed_edge_stance=(torch.arange(n - 1) % 3) - 1,
                root_index=torch.tensor([0]), y=torch.tensor([label]))


def model(**options):
    args = dict(dropout=0.0, dignn_wl_clusters=[5, 3], dignn_wl_buckets=128,
                dignn_wl_rounds=1, dignn_topic_clusters=4, max_hop=2)
    args.update(options)
    return DIGNN(4, 8, 2, SimpleNamespace(**args))


class SoftTopicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(93)

    def test_default_topics_and_backward_from_classification_only(self):
        network = model(dignn_topic_semantic_weight=0, dignn_topic_diversity_weight=0).train()
        self.assertEqual(network.text_mode, 'topics')
        batch = Batch.from_data_list([graph(), graph(4, 1), graph(1)])
        optimizer = network.init_optimizer()
        before = network.topic_encoder.slots.detach().clone()
        loss = network.classification_loss(network(batch)[0], batch.y)
        self.assertEqual(float(network.auxiliary_loss().detach()), 0)
        loss.backward()
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for part in ('slots', 'key.weight', 'query.weight', 'condition.weight',
                     'slot_update.weight_ih', 'semantic_q.weight', 'root_relation.0.weight',
                     'messages.0.weight', 'messages.1.weight', 'messages.4.weight'):
            grad = dict(network.topic_encoder.named_parameters())[part].grad
            self.assertGreater(float(grad.abs().sum()), 0, part)
        optimizer.step()
        self.assertFalse(torch.equal(before, network.topic_encoder.slots))

    def test_coarse_graph_matches_dense_reference_and_gradients(self):
        assignment = torch.tensor([[1., 0., 0.], [0., .8, .2],
                                   [0., .3, .7], [0., .1, .9]], requires_grad=True)
        edges = torch.tensor([[0, 1, 1], [1, 2, 3]])
        stance = torch.tensor([0, 1, -1])
        coarse = SoftTopicEncoder.coarse_edges(assignment, edges, stance)
        reference = []
        for kind in (0, 1, -1):
            dense = torch.zeros(4, 4)
            selected = edges[:, stance == kind]
            dense[selected[0], selected[1]] = 1
            reference.append(assignment.T @ dense @ assignment)
        torch.testing.assert_close(coarse, torch.stack(reference))
        torch.testing.assert_close(coarse.sum((1, 2)), torch.ones(3))
        self.assertFalse(torch.allclose(coarse, coarse.transpose(-1, -2)))
        coarse.square().sum().backward()
        self.assertGreater(float(assignment.grad[1:].abs().sum()), 0)

    def test_assignments_root_capacity_and_mass_conservation(self):
        network = model(dignn_topic_log_assignments=True).eval()
        data = Batch.from_data_list([graph(6), graph(2), graph(1)])
        views = network.encode_views(data)
        s = views['topic_assignment']
        expected = torch.ones(9)
        expected[data.root_index] = 0
        torch.testing.assert_close(s.sum(-1), expected)
        torch.testing.assert_close(views['topic_mass'].sum(-1), torch.tensor([5., 1., 0.]))
        torch.testing.assert_close(views['topic_mask'].sum(-1), torch.tensor([4, 1, 0]))
        torch.testing.assert_close(views['topic_edges'].sum((1, 2, 3)), torch.tensor([5., 1., 0.]))
        self.assertEqual(float(s[6:, 1:].detach().abs().sum()), 0)
        network(data)
        self.assertIn('topic_assignment', network.get_diagnostics())
        self.assertEqual(float(network.get_diagnostics()['topic_effective_count'][-1]), 0)

    def test_single_batch_event_order_and_node_permutation(self):
        network = model().eval()
        events = [graph(), graph(1), graph(3)]
        expected = torch.cat([network(g)[0] for g in events])
        torch.testing.assert_close(network(Batch.from_data_list(events))[0], expected)
        torch.testing.assert_close(network(Batch.from_data_list(events[::-1]))[0], expected.flip(0))
        original = events[0]
        order = torch.tensor([4, 1, 5, 0, 3, 2])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(6)
        permuted = original.clone()
        permuted.x = original.x[order]
        permuted.root_index = inverse[original.root_index]
        permuted.edge_index = inverse[original.edge_index].flip(1)
        permuted.directed_edge_index = inverse[original.directed_edge_index].flip(1)
        permuted.directed_edge_stance = original.directed_edge_stance.flip(0)
        torch.testing.assert_close(network(original)[0], network(permuted)[0])
        a, b = network.encode_views(original), network.encode_views(permuted)
        torch.testing.assert_close(a['topic_assignment'][order], b['topic_assignment'])
        torch.testing.assert_close(a['topic_edges'], b['topic_edges'])

    def test_semantic_grouping_is_independent_of_topology_but_interaction_is_not(self):
        network = model().eval()
        original = graph()
        original.x.requires_grad_(True)
        a = network.encode_views(original)
        changed = original.clone()
        changed.directed_edge_stance = (original.directed_edge_stance + 2) % 3 - 1
        b = network.encode_views(changed)
        torch.testing.assert_close(a['topic_assignment'], b['topic_assignment'])
        torch.testing.assert_close(a['text_graph'], b['text_graph'])
        self.assertFalse(torch.allclose(a['topic_graph'], b['topic_graph']))
        changed = original.clone()
        changed.x = changed.x * -7
        torch.testing.assert_close(a['topology_graph'], network.encode_views(changed)['topology_graph'])
        self.assertIsNone(torch.autograd.grad(a['topology_graph'].square().sum(),
                                              original.x, allow_unused=True)[0])
        expected = network(original)[0]
        del original.y
        original.user_state = torch.randn(1, 2, 3)
        original.node_state = torch.ones(6)
        torch.testing.assert_close(expected, network(original)[0])

    def test_edge_free_forest_unknown_stance_and_root_only_training(self):
        network = model().train()
        isolated = graph(4)
        isolated.directed_edge_index = isolated.edge_index = torch.empty(2, 0, dtype=torch.long)
        del isolated.directed_edge_stance
        forest = graph(5)
        forest.directed_edge_index = forest.edge_index = torch.tensor([[0, 3], [1, 4]])
        forest.directed_edge_stance = torch.tensor([-1, 1])
        for event in (isolated, forest, graph(1)):
            network.zero_grad()
            loss = network.compute_loss(event)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for p in network.parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
        network.eval()
        unknown = graph(3)
        del unknown.directed_edge_stance
        explicit = unknown.clone()
        explicit.directed_edge_stance = torch.full((2,), -1, dtype=torch.long)
        torch.testing.assert_close(network(unknown)[0], network(explicit)[0])

    def test_semantic_regularizer_uses_fixed_features_and_favors_semantic_pairs(self):
        encoder = model().topic_encoder
        raw = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        good = torch.tensor([[.95, .05], [.95, .05], [.05, .95], [.05, .95]], requires_grad=True)
        bad = torch.tensor([[.95, .05], [.05, .95], [.95, .05], [.05, .95]], requires_grad=True)
        good_loss, diversity = encoder._regularizers(good, raw)
        bad_loss, _ = encoder._regularizers(bad, raw)
        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        (good_loss + diversity).backward()
        self.assertIsNone(raw.grad)
        self.assertGreater(float(good.grad.abs().sum()), 0)
        network = model().train()
        data = graph(5)
        data.x[1:] = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.],
                                    [0., 1., 0., 0.], [0., 1., 0., 0.]])
        network(data)
        self.assertGreater(float(network.auxiliary_loss().detach()), 0)
        network.eval()
        network(data)
        self.assertEqual(float(network.auxiliary_loss().detach()), 0)

    def test_checkpoint_ablations_and_config_validation(self):
        data = graph()
        for options in ({}, {'dignn_topic_propagation': False},
                        {'dignn_topic_semantic_relations': False},
                        {'dignn_text_mode': 'mean'}, {'dignn_fusion': 'topology'}):
            network = model(**options).eval()
            expected = network(data)[0]
            stream = io.BytesIO()
            torch.save(network.state_dict(), stream)
            stream.seek(0)
            restored = model(**options).eval()
            restored.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(expected, restored(data)[0])
            if options.get('dignn_topic_propagation') is False:
                views = network.encode_views(data)
                torch.testing.assert_close(views['text_graph'], views['topic_graph'])
            if options.get('dignn_fusion') == 'topology':
                network.train()(data)
                self.assertEqual(float(network.auxiliary_loss().detach()), 0)
        for options in ({'dignn_topic_clusters': 0}, {'dignn_topic_steps': 0},
                        {'dignn_topic_temperature': float('nan')},
                        {'dignn_topic_semantic_weight': -1}, {'dignn_text_mode': 'invalid'},
                        {'dignn_text_mode': 'topics', 'dignn_topology_features': 'energy'}):
            with self.assertRaises(ValueError):
                model(**options)


if __name__ == '__main__':
    unittest.main()
