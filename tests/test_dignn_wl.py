"""Behavioral checks for the stance-WL classification-only topology branch."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

from model.DIGNN import DIGNN, StanceWLTokenizer


def graph(n, edges, stance=None, label=0):
    edges = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
    data = Data(x=torch.randn(n, 4), edge_index=edges.clone(),
                directed_edge_index=edges, root_index=torch.tensor([0]),
                y=torch.tensor([label]))
    if stance is not None:
        data.directed_edge_stance = torch.tensor(stance, dtype=torch.long)
    return data


def model(**options):
    settings = dict(dignn_topology_features="wl", dropout=0.0,
                    dignn_wl_clusters=[6, 3], dignn_wl_buckets=128,
                    dignn_wl_rounds=2, max_hop=3, dignn_text_mode="mean")
    settings.update(options)
    return DIGNN(4, 8, 2, SimpleNamespace(**settings))


class StanceWLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)

    def test_wl_counts_and_parent_relative_stance(self):
        tokenizer = StanceWLTokenizer(rounds=1)
        roots, batch = torch.tensor([0]), torch.zeros(3, dtype=torch.long)
        edges = torch.tensor([[0, 0], [1, 2]])
        same = tokenizer(edges, torch.tensor([0, 0]), batch, roots)
        mixed = tokenizer(edges, torch.tensor([0, 1]), batch, roots)
        # Same-stance children are one type; different-stance children split.
        self.assertEqual(sorted(same['counts'][same['scales'] == 1].tolist()), [1, 2])
        self.assertEqual(sorted(mixed['counts'][mixed['scales'] == 1].tolist()), [1, 1, 1])
        self.assertEqual(int(mixed['counts'].sum()), 6)
        # The direction of a relation matters, even when label counts match.
        chain = torch.tensor([[0, 1], [1, 2]])
        forward = tokenizer(chain, torch.tensor([0, 1]), batch, roots)
        reverse = tokenizer(chain, torch.tensor([1, 0]), batch, roots)
        self.assertFalse(torch.equal(forward['ids'], reverse['ids']))

    def test_wl_classification_only_and_finite_training_gradients(self):
        network = model(dropout=0.1, dignn_text_recon_weight=1,
                        dignn_structure_recon_weight=1, dignn_edge_recon_weight=1,
                        dignn_independence_weight=1).train()
        self.assertEqual(network.topology_features, 'wl')
        self.assertTrue(all(value == 0 for value in network.loss_weights.values()))
        data = Batch.from_data_list([
            graph(5, [[0, 0, 1, 1], [1, 2, 3, 4]], [0, 1, 0, 1]),
            graph(3, [[0, 1], [1, 2]], [1, -1], label=1),
            graph(1, [[], []], [])])
        optimizer = network.init_optimizer()
        before = network.topology_encoder.blocks[0].assignment[-1].weight.detach().clone()
        with (patch('model.DIGNN.normalized_hsic', side_effect=AssertionError('HSIC used')),
              patch('model.DIGNN.batched_negative_sampling', side_effect=AssertionError('reconstruction used'))):
            output = network(data)[0]
            loss = network.classification_loss(output, data.y) + network.auxiliary_loss()
        self.assertEqual(float(network.auxiliary_loss().detach()), 0)
        loss.backward()
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for module in (network.topology_encoder.blocks[0].attention,
                       network.topology_encoder.blocks[0].assignment,
                       network.topology_encoder.type_embeddings,
                       network.text_encoder, network.view_attention):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, network.topology_encoder.blocks[0].assignment[-1].weight))

    def test_batch_padding_event_order_and_node_order_invariance(self):
        network = model().eval()
        graphs = [graph(6, [[0, 0, 1, 2, 4], [1, 2, 3, 4, 5]], [0, 1, 1, -1, 0]),
                  graph(1, [[], []], []), graph(4, [[0], [1]], [-1])]
        output, u, s, d = network(Batch.from_data_list(graphs))
        single = torch.cat([network(g)[0] for g in graphs])
        torch.testing.assert_close(output, single, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(network(Batch.from_data_list(graphs[::-1]))[0], output.flip(0))
        self.assertEqual(u.shape, (3, 3, 1))
        self.assertEqual(s.shape, d.shape)
        torch.testing.assert_close(output.exp().sum(-1), torch.ones(3))
        original = graphs[0]
        order = torch.tensor([5, 2, 0, 3, 1, 4])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(6)
        permuted = original.clone()
        permuted.x = original.x[order]
        permuted.root_index = inverse[original.root_index]
        permuted.edge_index = inverse[original.edge_index].flip(1)
        permuted.directed_edge_index = inverse[original.directed_edge_index].flip(1)
        permuted.directed_edge_stance = original.directed_edge_stance.flip(0)
        torch.testing.assert_close(network(original)[0], network(permuted)[0])

    def test_text_stance_separation_and_no_target_leakage(self):
        network = model().eval()
        data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 1])
        data.x.requires_grad_(True)
        original = network.encode_views(data)
        self.assertIsNone(torch.autograd.grad(original['topology_graph'].sum(), data.x,
                                              allow_unused=True)[0])
        changed = data.clone()
        changed.x = -10 * changed.x
        torch.testing.assert_close(original['topology_graph'], network.encode_views(changed)['topology_graph'])
        changed = data.clone()
        changed.directed_edge_stance = torch.tensor([1, 0, 0])
        views = network.encode_views(changed)
        torch.testing.assert_close(original['text_graph'], views['text_graph'])
        self.assertFalse(torch.allclose(original['topology_graph'], views['topology_graph']))
        expected = network(data)[0]
        del data.y
        data.node_state = torch.ones(4)
        data.user_state = torch.randn(1, 3, 3)
        torch.testing.assert_close(expected, network(data)[0])

    def test_unknown_is_not_support_and_old_cache_alignment_is_preserved(self):
        network = model().eval()
        data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 1])
        old = Data(x=data.x, root_index=data.root_index,
                   edge_index=torch.tensor([[2, 3, 0, 1, 1, 0], [0, 1, 2, 3, 0, 1]]),
                   edge_stance=torch.tensor([1, 1, 1, 1, 0, 0]),
                   directed_edge_index=data.directed_edge_index.flip(1))
        torch.testing.assert_close(network(data)[0], network(old)[0])
        unknown = graph(4, [[0, 0, 1], [1, 2, 3]])
        explicit = unknown.clone()
        explicit.directed_edge_stance = torch.full((3,), -1, dtype=torch.long)
        torch.testing.assert_close(network(unknown)[0], network(explicit)[0])
        explicit.directed_edge_stance.zero_()
        self.assertFalse(torch.allclose(network.encode_views(unknown)['topology_graph'],
                                        network.encode_views(explicit)['topology_graph']))

    def test_mass_conservation_attention_and_padding(self):
        network = model(dignn_wl_log_attention=True).eval()
        data = Batch.from_data_list([graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 0]),
                                     graph(1, [[], []], [])])
        views = network.encode_views(data)
        torch.testing.assert_close(views['topology_token_mass'].sum(-1), torch.tensor([12., 3.]))
        valid = views['wl_tokens']['valid']
        diagnostics = views['wl_diagnostics']
        for i in range(2):
            assignment = diagnostics[f'wl_level_{i}_assignment']
            attention = diagnostics[f'wl_level_{i}_attention']
            torch.testing.assert_close(assignment.sum(-1), valid.float())
            torch.testing.assert_close(attention.sum(-1), valid.float())
            self.assertEqual(float(attention.masked_select(~valid[:, None, :]).abs().sum()), 0)
            count = diagnostics[f'wl_level_{i}_token_count']
            self.assertTrue(bool((count <= valid.sum(-1)).all()))
            valid = torch.arange(assignment.size(-1))[None] < count[:, None]
        self.assertTrue(bool(torch.isfinite(views['topology_graph']).all()))

    def test_checkpoint_unseen_graphs_and_cache_eviction_do_not_change_encoding(self):
        network = model(dignn_wl_cache_size=1).eval()
        first = graph(3, [[0, 1], [1, 2]], [0, 1])
        second = graph(5, [[0, 0, 0, 0], [1, 2, 3, 4]], [1, 0, -1, 1])
        expected = network(first)[0]
        network(second)
        torch.testing.assert_close(expected, network(first)[0])
        stream = io.BytesIO()
        torch.save(network.state_dict(), stream)
        stream.seek(0)
        restored = model(dignn_wl_cache_size=0).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(expected, restored(first)[0])
        torch.testing.assert_close(network(second)[0], restored(second)[0])

    def test_invalid_stances_and_cross_graph_edges_are_rejected(self):
        network = model()
        data = graph(3, [[0, 1], [1, 2]], [0, 1])
        for labels in (torch.tensor([0]), torch.tensor([0., 1.]), torch.tensor([0, 2])):
            data.directed_edge_stance = labels
            with self.assertRaises(ValueError):
                network(data)
        batch = Batch.from_data_list([graph(2, [[0], [1]], [0]), graph(2, [[0], [1]], [1])])
        batch.directed_edge_index[1, 0] = 2
        with self.assertRaisesRegex(ValueError, 'different events'):
            network(batch)


if __name__ == '__main__':
    unittest.main()
