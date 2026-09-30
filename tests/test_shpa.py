import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.SHPA import (
    SHPA,
    StanceBiGCN,
    StanceGCN,
    StanceResGCN,
    channel_contrastive_loss,
)


def graph(label=0, n=4):
    return Data(
        x=torch.randn(n, 5),
        edge_index=torch.stack((torch.arange(n - 1), torch.arange(1, n))),
        edge_stance=torch.arange(n - 1) % 2,
        y=torch.tensor([label]),
        user_state=torch.zeros(1, 3, 3),
    )


class SHPATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.args = SimpleNamespace(max_hop=3, shpa_lambda_contrastive=2.0,
                                    lr=0.001, weight_decay=0.0)
        self.model = SHPA(5, 8, 2, self.args)

    def test_contrastive_matches_equation_3_and_gradients(self):
        support = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
        deny = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
        channels = [F.normalize(support, dim=-1), F.normalize(deny, dim=-1)]
        terms = []
        for c in range(2):
            for i in range(3):
                denominator = sum(torch.exp(channels[c][i].dot(channels[c][k]) / 0.2)
                                  for k in range(3) if k != i)
                denominator += sum(torch.exp(channels[c][i].dot(channels[1-c][k]) / 0.2)
                                   for k in range(3))
                for j in range(3):
                    if j != i:
                        terms.append(-(channels[c][i].dot(channels[c][j]) / 0.2
                                       - denominator.log()))
        expected = torch.stack(terms).mean()
        actual = channel_contrastive_loss(support, deny)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, (support, deny), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (support, deny))
        for a, e in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, e)

    def test_contrastive_equal_vectors_and_singleton(self):
        self.assertAlmostEqual(channel_contrastive_loss(torch.ones(3, 4),
                                                        torch.ones(3, 4)).item(),
                               math.log(5), places=6)
        x = torch.randn(1, 4, requires_grad=True)
        loss = channel_contrastive_loss(x, -x)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_partition_is_undirected_and_deduplicated(self):
        data = Data(x=torch.randn(3, 5),
                    edge_index=torch.tensor([[0, 1, 1, 0], [1, 0, 2, 1]]),
                    edge_stance=torch.tensor([0, 0, 1, 0]))
        support, deny = self.model.stance_graphs(data)
        self.assertEqual(set(map(tuple, support.t().tolist())), {(0, 1), (1, 0)})
        self.assertEqual(set(map(tuple, deny.t().tolist())), {(1, 2), (2, 1)})
        self.assertEqual(support.size(1), 2)

    def test_legacy_cache_uses_matching_edges_and_labels(self):
        data = graph()
        expected = self.model(data)[0]
        # Old caches can have directed_edge_index but no directed_edge_stance.
        data.directed_edge_index = data.edge_index.flip(1)
        torch.testing.assert_close(self.model(data)[0], expected)

    def test_training_interface_and_both_channel_gradients(self):
        data = Batch.from_data_list([graph(0), graph(1, 5)])
        out, u, s, d = self.model(data)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(u.shape, (2, 3, 1))
        torch.testing.assert_close(out.exp().sum(-1), torch.ones(2))
        loss = (self.model.classification_loss(out, data.y)
                + self.model.physics_loss(u, s, d, data.user_state)
                + self.model.auxiliary_loss())
        loss.backward()
        for encoder in (self.model.support_encoder, self.model.deny_encoder):
            grads = [p.grad for p in encoder.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.model.init_optimizer(self.args).step()

    def test_all_configurable_backbones_forward_and_backpropagate(self):
        expected_types = {
            'gcn': StanceGCN,
            'resgcn': StanceResGCN,
            'bigcn': StanceBiGCN,
        }
        data = Batch.from_data_list([graph(0), graph(1, 5)])
        for backbone, expected_type in expected_types.items():
            with self.subTest(backbone=backbone):
                args = SimpleNamespace(
                    max_hop=3,
                    shpa_backbone=backbone,
                    shpa_num_layers=3,
                    shpa_lambda_contrastive=1.0,
                    dropout=0.0,
                    edge_norm=True,
                    lr=0.001,
                    weight_decay=0.0,
                )
                model = SHPA(5, 8, 2, args, torch.device('cpu'))
                self.assertIsInstance(model.support_encoder, expected_type)
                self.assertIsInstance(model.deny_encoder, expected_type)
                output = model(data)[0]
                loss = model.classification_loss(output, data.y)
                loss = loss + model.auxiliary_loss()
                loss.backward()
                self.assertEqual(output.shape, (2, 2))
                self.assertTrue(torch.isfinite(output).all())
                for encoder in (model.support_encoder, model.deny_encoder):
                    gradients = [parameter.grad for parameter in encoder.parameters()]
                    self.assertTrue(gradients)
                    self.assertTrue(all(g is not None for g in gradients))
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_invalid_backbone_is_rejected(self):
        args = SimpleNamespace(max_hop=3, shpa_backbone='transformer')
        with self.assertRaisesRegex(ValueError, 'gcn, resgcn, bigcn'):
            SHPA(5, 8, 2, args)

    def test_eval_skips_alignment_and_is_batch_independent(self):
        a, b = graph(), graph(1)
        self.model(Batch.from_data_list([a, b]))
        self.model.eval()
        self.assertEqual(self.model.auxiliary_loss().item(), 0)
        with patch('model.SHPA.channel_contrastive_loss', side_effect=AssertionError):
            single = self.model(a)[0]
            batched = self.model(Batch.from_data_list([a, b]))[0]
        torch.testing.assert_close(single[0], batched[0])
        self.assertIsNone(self.model._contrastive_loss)

    def test_root_only_and_missing_relation_are_finite(self):
        for n in (1, 4):
            data = graph(n=n)
            data.edge_stance.zero_()
            out = self.model(data)[0]
            self.assertTrue(torch.isfinite(out).all())
            (out.sum() + self.model.auxiliary_loss()).backward()

    def test_invalid_stances_fail_instead_of_silently_changing_graph(self):
        for label in (-1, 2, 0.5):
            data = graph()
            data.edge_stance = data.edge_stance.float()
            data.edge_stance[0] = label
            with self.assertRaisesRegex(ValueError, 'complete 0/1'):
                self.model(data)
        data = graph()
        del data.edge_stance
        with self.assertRaisesRegex(ValueError, 'requires LLM'):
            self.model(data)

    def test_no_contrastive_ablation(self):
        self.model.lambda_contrastive = 0.0
        with patch('model.SHPA.channel_contrastive_loss', side_effect=AssertionError):
            self.model(Batch.from_data_list([graph(), graph(1)]))
        self.assertEqual(self.model.auxiliary_loss().item(), 0)

    def test_cache_key_matches_existing_resgcn_cache(self):
        from supervisor import get_dataset_cache_name

        args = SimpleNamespace(
            base_model='SHPA', experiment_mode='id', word_embedding='word2vec',
            language='en', max_hop=47, centrality='PageRank', undirected=False,
            tokenize_mode='nltk', vector_size=200,
        )
        shpa_key = get_dataset_cache_name(args)
        args.base_model = 'ResGCN'
        self.assertEqual(shpa_key, get_dataset_cache_name(args))

    def test_cached_supervisor_does_not_build_text_encoder(self):
        import supervisor

        args = SimpleNamespace(
            seed=0, device='cpu', dataset='Pheme', base_model='SHPA',
            experiment_mode='id', word_embedding='word2vec', language='en',
            max_hop=47, centrality='PageRank', undirected=False,
            tokenize_mode='nltk', vector_size=200, split='622', k=10000,
            in_feats=5, hidden_dim=8, num_classes=2,
        )

        class DummyModel:
            def to(self, device):
                return self

            def init_optimizer(self, model_args):
                return object()

        class DummyTrainer:
            def __init__(self, datasets, model, optimizer, trainer_args, device):
                self.datasets = datasets

            def train_process(self):
                return self.datasets

        cached = (object(), object(), object())
        with patch.object(supervisor, 'init_seed'), \
                patch.object(supervisor, 'resolve_device', return_value=torch.device('cpu')), \
                patch.object(supervisor, 'load_cached_experiment_datasets', return_value=cached), \
                patch.object(supervisor, 'build_text_encoder', side_effect=AssertionError), \
                patch.object(supervisor, 'SHPA', return_value=DummyModel()), \
                patch.object(supervisor, 'EINTrainer', DummyTrainer):
            self.assertIs(supervisor.EIN_SHPA_supervisor(args), cached)


if __name__ == '__main__':
    unittest.main()
