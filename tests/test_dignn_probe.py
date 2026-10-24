"""Numerical and end-to-end checks for the structural probing topology branch."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

from model.DIGNN import DIGNN, StanceStructureProbe


def graph(n, edges, stance, label=0):
    edges = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
    return Data(x=torch.randn(n, 4), edge_index=edges.clone(),
                directed_edge_index=edges, directed_edge_stance=torch.tensor(stance, dtype=torch.long),
                root_index=torch.tensor([0]), y=torch.tensor([label]))


def model(**options):
    args = dict(dropout=0.0, max_hop=3, dignn_text_mode="mean")
    args.update(options)
    return DIGNN(4, 8, 2, SimpleNamespace(**args))


def dense_reference(data, scales):
    n = data.num_nodes
    source = torch.zeros(n, dtype=torch.float64)
    source[data.root_index] = 1
    generators = []
    for relation in (None, 0, 1):
        edges = data.directed_edge_index
        if relation is not None:
            edges = edges[:, data.directed_edge_stance == relation]
        adjacency = torch.zeros(n, n, dtype=torch.float64)
        adjacency[edges[0], edges[1]] = 1
        generators.append(adjacency.T - torch.diag(adjacency.sum(1)))
    result = []
    for scale in scales:
        full, support, deny = [torch.matrix_exp(scale * q) for q in generators]
        sd, ds = deny @ support @ source, support @ deny @ source
        result.append(torch.stack((full @ source, support @ source, deny @ source,
                                   sd, ds, sd - ds), -1))
    return torch.stack(result, 1)


class StructureProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(107)

    def test_sparse_action_matches_independent_dense_matrix_exponential(self):
        data = graph(8, [[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]], [0, 1, 1, -1, 0, 1])
        scales = (0.03, 0.25, 1.0, 4.0)
        probe = StanceStructureProbe(8, 0.0, SimpleNamespace(dignn_probe_scales=scales)).double()
        expected = dense_reference(data, scales)
        with patch('torch.matrix_exp', side_effect=AssertionError('dense exponential used')):
            result = probe(data.directed_edge_index, data.directed_edge_stance,
                           torch.zeros(8, dtype=torch.long), data.root_index)['probe_responses']
        torch.testing.assert_close(result, expected, atol=2e-12, rtol=2e-12)
        self.assertGreaterEqual(float(result[..., :5].min()), -1e-14)
        torch.testing.assert_close(result[..., :5].sum(0), torch.ones(4, 5, dtype=torch.float64))
        torch.testing.assert_close(result[..., 5].sum(0), torch.zeros(4, dtype=torch.float64),
                                   atol=1e-13, rtol=0)
        self.assertEqual(float(result[7].abs().sum()), 0)  # Isolate is not a source.

    def test_order_sensitivity_with_identical_shape_and_per_depth_counts(self):
        # A: ss, sd, dd; B: sd, sd, ds. Same topology and same S/D per depth.
        edges = [[0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 5, 6]]
        a, b = graph(7, edges, [0, 0, 1, 0, 1, 1]), graph(7, edges, [0, 0, 1, 1, 1, 0])
        b.x = a.x.clone()
        network = model(dignn_probe_scales=[1.0]).eval()
        va, vb = network.encode_views(a), network.encode_views(b)
        ra, rb = va['probe_responses'], vb['probe_responses']
        torch.testing.assert_close(ra[..., 0], rb[..., 0])
        self.assertAlmostEqual(float(ra[..., 5].square().sum()), 0.266, places=3)
        self.assertAlmostEqual(float(rb[..., 5].square().sum()), 0.331, places=3)
        self.assertFalse(torch.allclose(va['topology_graph'], vb['topology_graph']))

    def test_unknown_edges_and_alternating_paths_use_full_operator(self):
        network = model().eval()
        chain = graph(5, [[0, 1, 2, 3], [1, 2, 3, 4]], [0, 1, 0, 1])
        responses = network.encode_views(chain)['probe_responses']
        self.assertTrue(bool((responses[-1, :, 0] > 0).all()))
        self.assertEqual(float(responses[-1, :, 1:].abs().sum()), 0)
        unknown = graph(3, [[0, 1], [1, 2]], [-1, -1])
        responses = network.encode_views(unknown)['probe_responses']
        self.assertTrue(bool((responses[-1, :, 0] > 0).all()))
        self.assertEqual(float(responses[1:, :, 1:].abs().sum()), 0)
        missing = unknown.clone()
        del missing.directed_edge_stance
        torch.testing.assert_close(network(unknown)[0], network(missing)[0])
        unknown.directed_edge_stance.zero_()
        self.assertFalse(torch.allclose(network.encode_views(unknown)['probe_responses'], responses))

    def test_text_and_labels_do_not_enter_probes_and_existing_topics_are_preserved(self):
        for text_mode in ('mean', 'topics'):
            network = model(dignn_text_mode=text_mode).eval()
            data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, -1])
            data.x.requires_grad_(True)
            with patch.object(DIGNN, '_structural_features', side_effect=AssertionError('degree features used')):
                views = network.encode_views(data)
            self.assertIsNone(torch.autograd.grad(views['topology_graph'].square().sum(), data.x,
                                                  allow_unused=True)[0])
            changed = data.clone()
            changed.x = -7 * changed.x
            del changed.y
            changed.node_state = torch.ones(4)
            changed.user_state = torch.randn(1, 3, 3)
            torch.testing.assert_close(views['topology_graph'], network.encode_views(changed)['topology_graph'])
            # Copy exactly the shared text/topic weights from probe to old WL.
            previous = model(dignn_topology_features='wl', dignn_text_mode=text_mode).eval()
            state = previous.state_dict()
            shared = {k: v for k, v in network.state_dict().items()
                      if k.startswith(('text_encoder.', 'text_readout.', 'topic_encoder.'))}
            state.update(shared)
            previous.load_state_dict(state)
            old_views = previous.encode_views(data)
            for name in ('text_nodes', 'text_graph') + (('topic_graph',) if text_mode == 'topics' else ()):
                torch.testing.assert_close(views[name], old_views[name])

    def test_batch_event_and_node_permutation_invariance(self):
        network = model().eval()
        graphs = [graph(6, [[0, 0, 1, 2, 4], [1, 2, 3, 4, 5]], [0, 1, 1, -1, 0]),
                  graph(1, [[], []], []), graph(4, [[1], [2]], [1])]
        output = network(Batch.from_data_list(graphs))[0]
        torch.testing.assert_close(output, torch.cat([network(g)[0] for g in graphs]), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(network(Batch.from_data_list(graphs[::-1]))[0], output.flip(0))
        original = graphs[0]
        order = torch.tensor([5, 2, 0, 3, 1, 4])
        inverse = torch.argsort(order)
        permuted = original.clone()
        permuted.x = original.x[order]
        permuted.root_index = inverse[original.root_index]
        permuted.edge_index = inverse[original.edge_index].flip(1)
        permuted.directed_edge_index = inverse[original.directed_edge_index].flip(1)
        permuted.directed_edge_stance = original.directed_edge_stance.flip(0)
        a, b = network.encode_views(original), network.encode_views(permuted)
        torch.testing.assert_close(a['probe_responses'][order], b['probe_responses'])
        torch.testing.assert_close(a['topology_nodes'][order], b['topology_nodes'])
        torch.testing.assert_close(network(original)[0], network(permuted)[0])

    def test_forward_backward_optimizer_and_diagnostics(self):
        for mode in ('mean', 'topics'):
            network = model(dignn_text_mode=mode, dignn_topic_semantic_weight=0,
                            dignn_topic_diversity_weight=0).train()
            self.assertEqual(network.topology_features, 'probe')
            data = Batch.from_data_list([
                graph(5, [[0, 0, 1, 2], [1, 2, 3, 4]], [0, 1, 1, 0]),
                graph(3, [[0, 1], [1, 2]], [1, -1], label=1), graph(1, [[], []], [])])
            optimizer = network.init_optimizer()
            before = network.topology_encoder.node_encoder[0].weight.detach().clone()
            output, u, s, d = network(data)
            self.assertEqual(output.shape, (3, 2))
            self.assertEqual(u.shape, (3, 3, 1))
            self.assertEqual(s.shape, d.shape)
            torch.testing.assert_close(output.exp().sum(-1), torch.ones(3))
            self.assertEqual(float(network.auxiliary_loss().detach()), 0)
            network.classification_loss(output, data.y).backward()
            for name, parameter in network.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(network.topology_encoder.node_encoder[0].weight.grad.abs().sum()), 0)
            optimizer.step()
            self.assertFalse(torch.equal(before, network.topology_encoder.node_encoder[0].weight))
            diagnostics = network.get_diagnostics()
            torch.testing.assert_close(diagnostics['probe_mass'][..., :5], torch.ones(3, 3, 5))
            torch.testing.assert_close(diagnostics['probe_mass'][..., 5], torch.zeros(3, 3), atol=1e-7, rtol=0)
            self.assertTrue(all(not v.requires_grad for v in diagnostics.values()))

    def test_ablations_retain_the_matching_response_channels(self):
        data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, -1, 1])
        reference = model().encode_views(data)['probe_responses']
        for mode, width in (('untyped', 1), ('typed', 3), ('ordered', 6)):
            network = model(dignn_probe_mode=mode).eval()
            views = network.encode_views(data)
            torch.testing.assert_close(views['probe_responses'], reference[..., :width])
            self.assertTrue(bool(torch.isfinite(network(data)[0]).all()))

    def test_old_cache_alignment_conflicts_and_invalid_graphs(self):
        network = model().eval()
        data = graph(3, [[0, 1], [1, 2]], [0, 1])
        old = Data(x=data.x, root_index=data.root_index,
                   directed_edge_index=data.directed_edge_index.flip(1),
                   edge_index=torch.tensor([[2, 0, 1, 1], [1, 1, 0, 2]]),
                   edge_stance=torch.tensor([1, 0, 0, 1]))
        torch.testing.assert_close(network(data)[0], network(old)[0])
        for bad in (graph(2, [[0, 0], [1, 1]], [0, 1]),
                    graph(3, [[0, 1, 2], [1, 2, 1]], [0, 1, 0]),
                    graph(2, [[0], [1]], [2])):
            with self.assertRaises(ValueError):
                network(bad)
        cross = Batch.from_data_list([data, data])
        cross.directed_edge_index[1, 0] = 3
        with self.assertRaisesRegex(ValueError, 'different events'):
            network(cross)

    def test_cache_checkpoint_and_changed_scale_buffers(self):
        network = model(dignn_probe_cache_size=1).eval()
        data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, -1, 1])
        expected = network(data)[0]
        network(graph(1, [[], []], []))
        self.assertEqual(len(network.topology_encoder._cache), 1)
        torch.testing.assert_close(expected, network(data)[0])
        stream = io.BytesIO()
        torch.save(network.state_dict(), stream)
        stream.seek(0)
        restored = model(dignn_probe_cache_size=0).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(expected, restored(data)[0])
        self.assertEqual(len(restored.topology_encoder._cache), 0)
        other = model(dignn_probe_scales=[0.1, 0.2, 0.3]).eval()
        network.load_state_dict(other.state_dict())  # Existing cache has the old scales.
        torch.testing.assert_close(network(data)[0], other(data)[0])

    def test_high_degree_star_matches_analytic_mass(self):
        n = 257
        data = graph(n, [([0] * (n - 1)), list(range(1, n))], [0] * (n - 1))
        network = model().eval()
        responses = network.encode_views(data)['probe_responses']
        scales = network.topology_encoder.scales.float()
        root = torch.exp(-(n - 1) * scales)
        torch.testing.assert_close(responses[0, :, 0], root, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(responses[1:, :, 0],
                                   ((1 - root) / (n - 1)).expand(n - 1, -1))
        self.assertTrue(bool(torch.isfinite(responses).all()))

    def test_invalid_configuration(self):
        for options in ({'dignn_probe_scales': []}, {'dignn_probe_scales': [0]},
                        {'dignn_probe_scales': [float('nan')]}, {'dignn_probe_scales': None},
                        {'dignn_probe_mode': 'invalid'}, {'dignn_probe_cache_size': -1},
                        {'dignn_probe_cache_size': 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                model(**options)
        self.assertEqual(DIGNN(4, 8).text_mode, 'topics')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_matches_cpu(self):
        network = model().eval()
        data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, -1])
        expected = network(data)[0]
        torch.testing.assert_close(network.cuda()(data.cuda())[0].cpu(), expected, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
