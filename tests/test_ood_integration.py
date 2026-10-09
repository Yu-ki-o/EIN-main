"""OOD manifests through the existing graph loaders and model supervisors.

All data and caches live in temporary directories. Only text encoding and the
outer training loop are replaced; graph preprocessing, model forward/backward,
optimizer updates, and manifest validation are real.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import yaml

import supervisor
from trainer.EIN_trainer import EINTrainer
from utils.ood_runtime import prepare_ood
from utils.ood_splits import build_ood_manifest, materialize_ood_posts, write_ood_manifest


ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder:
    embedding_dim = 8

    def get_sentence_embeddings(self, texts):
        return torch.tensor([
            [value / 255 for value in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ], dtype=torch.float32)


class OODIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.original_cwd = Path.cwd()
        os.chdir(self.directory)
        self.dataset_root = self.directory / 'dataset'
        for dataset in ('DRWeibo', 'Weibo', 'Pheme'):
            source = self.dataset_root / dataset / 'source'
            source.mkdir(parents=True)
            for index in range(10):
                identifier = '{}_{}'.format(dataset, index)
                post = {
                    'source': {'tweet id': identifier, 'content': identifier + ' root', 'label': index % 2},
                    'comment': [
                        {'comment id': 0, 'parent': -1, 'content': identifier + ' first', 'state': 0, 'stance_label': 0},
                        {'comment id': 1, 'parent': 0, 'content': identifier + ' second', 'state': 1, 'stance_label': 1},
                    ],
                    'state': {'1-1': {'state_0': 1, 'state_1': 0}, '2-2': {'state_0': 1, 'state_1': 1}},
                }
                (source / (identifier + '.json')).write_text(json.dumps(post), encoding='utf-8')
        self.manifest, self.path = self.make_manifest('DRWeibo', 'Pheme')

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temporary.cleanup()

    def make_manifest(self, source, target, seed=0, val_ratio=0.2):
        manifest = build_ood_manifest(
            dataset_root=self.dataset_root, protocol='cross_dataset',
            sources=[source], target=target, seed=seed, val_ratio=val_ratio, workers=1,
        )
        path = self.directory / '{}_{}_{}_{}.json'.format(source, target, seed, val_ratio)
        write_ood_manifest(manifest, path)
        return manifest, path

    def args(self, **overrides):
        config = yaml.safe_load((ROOT / 'configs/EIN/Pheme_strict_ood_DRWeibo_Weibo.yaml').read_text())
        config.update(
            experiment_mode='ood', ood_manifest=str(self.path), device='cpu', seed=0,
            hidden_dim=8, batch_size=4, n_layers_conv=2, n_layers_fc=1,
            n_layers_feat=1, dropout=0.0, in_feats=8, _ood_embedding_dim=8,
            max_hop=20, selection_metric='val_loss', debug=False,
        )
        config.update(overrides)
        return SimpleNamespace(**config)

    def test_real_loaders_preserve_manifest_splits_and_reuse_processed_cache(self):
        args = self.args()
        with patch.object(supervisor, 'build_split_posts', side_effect=AssertionError('ID splitting is forbidden')):
            datasets = supervisor.build_experiment_datasets(args, TinyEncoder())
            cached = supervisor.build_experiment_datasets(args, None)
        for split, actual, reloaded in zip(('train', 'val', 'test'), datasets, cached):
            posts = materialize_ood_posts(self.manifest, split)
            self.assertEqual(len(actual), len(posts))
            self.assertEqual(Counter(int(graph.y.item()) for graph in actual),
                             Counter(post['source']['label'] for _, post in posts))
            self.assertEqual(set(actual.raw_file_names), {identifier + '.json' for identifier, _ in posts})
            for index in range(len(actual)):
                graph = actual[index]
                self.assertEqual(tuple(graph.user_state.shape), (1, 72, 3))
                self.assertEqual(int(graph.domain_id.item()), -1 if split == 'test' else 0)
                torch.testing.assert_close(graph.x, reloaded[index].x)
                self.assertEqual(graph.num_nodes, 3)
                self.assertEqual(graph.directed_edge_index.size(1), 2)
        _, cache_root = supervisor.dataset_paths(args, args.dataset)
        self.assertTrue((Path(cache_root) / 'ood_manifest.json').is_file())

    def test_cache_key_is_stable_before_and_after_preparation_and_binds_manifest(self):
        args = self.args()
        initial = supervisor.get_dataset_cache_name(args)
        self.assertEqual(initial, supervisor.get_dataset_cache_name(args))
        self.assertIn('hop-72', initial)
        self.assertIn('manifest-', initial)
        _, other_path = self.make_manifest('DRWeibo', 'Pheme', val_ratio=0.4)
        other = self.args(ood_manifest=str(other_path))
        self.assertNotEqual(initial, supervisor.get_dataset_cache_name(other))
        self.assertNotEqual(initial, supervisor.get_dataset_cache_name(self.args(experiment_mode='id')))

    def test_source_word2vec_sees_only_manifest_training_texts(self):
        manifest, path = self.make_manifest('DRWeibo', 'Weibo')
        args = self.args(dataset='Weibo', ood_manifest=str(path), word_embedding='word2vec',
                         vector_size=8, language='ch', tokenize_mode='jieba')
        training_texts = [node['content'] for _, post in materialize_ood_posts(manifest, 'train')
                          for node in [post['source']] + post['comment']]
        held_out_texts = {node['content'] for split in ('val', 'test')
                         for _, post in materialize_ood_posts(manifest, split)
                         for node in [post['source']] + post['comment']}
        fake_model = Mock()
        fake_model.save.side_effect = lambda destination: Path(destination).write_text('test word vectors')
        tokenizer = lambda text, **kwargs: [text]
        with patch.object(supervisor, 'word_tokenizer', side_effect=tokenizer), \
             patch.object(supervisor, 'train_word2vec', return_value=fake_model) as train, \
             patch.object(supervisor, 'Embedding', return_value=TinyEncoder()), \
             patch.object(supervisor, 'collect_sentences', side_effect=AssertionError('Full corpus tokenization is forbidden')):
            supervisor.build_text_encoder(args, torch.device('cpu'), 'unused-target-source')
            supervisor.build_text_encoder(args, torch.device('cpu'), 'unused-target-source')
        train.assert_called_once()
        seen = [sentence[0] for sentence in train.call_args.args[0]]
        self.assertCountEqual(seen, training_texts)
        self.assertFalse(set(seen) & held_out_texts)
        self.assertIn('word2vec/ood/', fake_model.save.call_args.args[0])

    def test_invalid_ood_options_fail_before_encoder_or_ready_cache(self):
        cases = [
            {'ood_val_domain': 'target'}, {'early_test_root': 'early'},
            {'p2t3_pretrained_path': 'old.pt'}, {'kpg_checkpoint': 'old.pt'},
            {'checkpoint_path': 'old.pt'}, {'see_ttt_enabled': True},
            {'base_model': 'SEEGraphMAE'}, {'kpg_test_only': True},
            {'word_embedding': 'word2vec', 'language': 'ch'},
        ]
        for overrides in cases:
            shpa_overrides = {k: v for k, v in overrides.items() if k != 'base_model'}
            if overrides.get('base_model') == 'SEEGraphMAE':
                shpa_overrides['see_ttt_enabled'] = True
            with self.subTest(overrides=overrides), \
                 patch.object(supervisor, 'MultilingualE5Embedding', side_effect=AssertionError('Encoder must not load')), \
                 patch.object(supervisor, 'load_cached_experiment_datasets', side_effect=AssertionError('Cache must not load')):
                with self.assertRaises(ValueError):
                    supervisor.build_text_encoder(self.args(**overrides), torch.device('cpu'), 'unused')
                with self.assertRaises(ValueError):
                    supervisor.EIN_SHPA_supervisor(self.args(base_model='SHPA', **shpa_overrides))

    def test_word2vec_rejects_external_vectors_even_for_same_language(self):
        _, path = self.make_manifest('DRWeibo', 'Weibo')
        with self.assertRaisesRegex(ValueError, 'word2vec_model_path'):
            prepare_ood(self.args(dataset='Weibo', ood_manifest=str(path), word_embedding='word2vec',
                                  word2vec_model_path='old.model', language='ch'))

    def test_existing_backbones_and_uncertainty_supervisors_train_on_real_ood_graphs(self):
        for model_name in ('BiGCN', 'ResGCN', 'BiGCN_Uncertainty', 'ResGCN_Uncertainty'):
            with self.subTest(model=model_name):
                args = self.args(base_model=model_name)
                factory = Mock()
                factory.return_value.train_process.return_value = {'acc': 0.5}
                with patch.object(supervisor, 'MultilingualE5Embedding', return_value=TinyEncoder()), \
                     patch.object(supervisor, 'EINTrainer', factory):
                    result = getattr(supervisor, 'EIN_' + model_name + '_supervisor')(args)
                self.assertEqual(result, {'acc': 0.5})
                datasets, model, optimizer, received_args, device = factory.call_args.args
                self.assertIs(received_args, args)
                batch = Batch.from_data_list([datasets[0][index] for index in range(4)])
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                output = model(batch)
                self.assertEqual(tuple(output[0].shape), (4, 2))
                loss = F.nll_loss(output[0], batch.y)
                loss = loss + model.physics_loss(*output[1:], batch.user_state)
                if hasattr(model, 'auxiliary_loss'):
                    loss = loss + model.auxiliary_loss()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                optimizer.step()
                self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()))

    def test_shpa_ready_cache_restores_feature_width_without_encoder(self):
        supervisor.build_experiment_datasets(self.args(base_model='ResGCN'), TinyEncoder())
        args = self.args(base_model='SHPA', in_feats=200)
        del args._ood_embedding_dim
        with patch.object(supervisor, 'build_text_encoder', side_effect=AssertionError('Ready cache should skip encoder')), \
             patch.object(supervisor, 'SHPA') as model_factory, \
             patch.object(supervisor, 'EINTrainer') as trainer:
            trainer.return_value.train_process.return_value = {'acc': 0.5}
            supervisor.EIN_SHPA_supervisor(args)
        self.assertEqual(model_factory.call_args.args[0], 8)
        self.assertEqual(args.in_feats, 8)

    def test_ein_auc_uses_probability_ranking_when_argmax_predictions_are_equal(self):
        dataset = supervisor.build_experiment_datasets(self.args(), TinyEncoder())[0]
        graphs = [next(graph for graph in dataset if int(graph.y.item()) == label) for label in (0, 1)]
        batch = Batch.from_data_list(graphs)
        trainer = EINTrainer.__new__(EINTrainer)
        trainer.model = Mock(return_value=(torch.tensor([[0.9, 0.1], [0.8, 0.2]]).log(), None, None, None))
        trainer.test_loader = [batch]
        trainer._move_to_device = lambda data: data
        trainer._accumulate_diagnostics = Mock()
        trainer._average_diagnostics = Mock(return_value={})
        trainer.logger = Mock()
        trainer.tb_writer = None
        metrics = trainer.test()
        self.assertEqual(metrics['acc'], 0.5)
        self.assertEqual(metrics['auc'], 1.0)


if __name__ == '__main__':
    unittest.main()
