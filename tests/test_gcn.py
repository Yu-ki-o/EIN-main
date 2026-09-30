import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch, Data

from model.GCN import GCN


class GCNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.args = SimpleNamespace(n_layers_conv=2, dropout=0.0,
                                    global_pool='mean', max_hop=3,
                                    lr=0.001, weight_decay=0.0001)
        self.model = GCN(5, 8, 2, self.args)
        self.graphs = [
            Data(x=torch.randn(3, 5),
                 edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                 y=torch.tensor([0]), user_state=torch.zeros(1, 3, 3)),
            Data(x=torch.randn(1, 5), edge_index=torch.empty(2, 0, dtype=torch.long),
                 y=torch.tensor([1]), user_state=torch.zeros(1, 3, 3)),
        ]

    def test_batch_matches_individual_graphs_including_edgeless_singleton(self):
        self.model.eval()
        together = self.model(Batch.from_data_list(self.graphs))[0]
        separately = torch.cat([self.model(graph)[0] for graph in self.graphs])
        torch.testing.assert_close(together, separately)
        torch.testing.assert_close(together.exp().sum(-1), torch.ones(2))

    def test_classification_training_and_stance_independence(self):
        batch = Batch.from_data_list(self.graphs)
        output, u, s, d = self.model(batch)
        self.assertEqual(u.shape, (2, 3, 1))
        physics = self.model.physics_loss(u, s, d, batch.user_state)
        self.assertEqual(physics.item(), 0.0)
        batch.user_state.fill_(100)
        batch.edge_stance = torch.ones(batch.edge_index.size(1))
        torch.testing.assert_close(output, self.model(batch)[0])
        before = self.model.classifier.weight.detach().clone()
        (F.nll_loss(output, batch.y) + physics).backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.model.init_optimizer(self.args).step()
        self.assertFalse(torch.equal(before, self.model.classifier.weight))

    def test_three_configs_and_supervisor_dispatch(self):
        import main
        import supervisor

        root = Path(__file__).resolve().parents[1]
        for dataset in ('Pheme', 'Weibo', 'DRWeibo'):
            with self.subTest(dataset=dataset):
                with (root / 'configs' / 'EIN' / f'{dataset}_GCN_word2vec.yaml').open() as handle:
                    args = SimpleNamespace(**yaml.safe_load(handle))
                self.assertEqual(args.dataset, dataset)
                self.assertEqual(supervisor._graph_dataset_cache_part(args), 'resgcn-tree')
                self.assertFalse(supervisor._requires_ragcl_centrality(args))
                self.assertIn('undir-True', supervisor.get_dataset_cache_name(args))
                self.assertIs(getattr(main, 'EIN_' + args.base_model + '_supervisor'),
                              supervisor.EIN_Plain_GCN_supervisor)
                args.device = 'cpu'
                datasets = ([self.graphs[0]], [self.graphs[1]], self.graphs)
                with patch.object(supervisor, 'build_text_encoder'), \
                     patch.object(supervisor, 'build_experiment_datasets', return_value=datasets), \
                     patch.object(supervisor, 'EINTrainer') as trainer:
                    supervisor.EIN_Plain_GCN_supervisor(args)
                    model = trainer.call_args.args[1]
                    self.assertIsInstance(model, GCN)
                    data = Data(x=torch.randn(1, args.in_feats),
                                edge_index=torch.empty(2, 0, dtype=torch.long))
                    self.assertEqual(model(data)[0].shape, (1, 2))
                    trainer.return_value.train_process.assert_called_once()


if __name__ == '__main__':
    unittest.main()
