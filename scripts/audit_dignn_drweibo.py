#!/usr/bin/env python3
"""Verify minimum-val_loss selection and native metrics on complete cached graphs."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import yaml

from tune_dignn_pheme import ROOT, cache_root
from model.DIGNN import DIGNN


def complete_graphs(path):
    data, slices = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    return [separate(data.__class__, data, i, slices, decrement=False)
            for i in range(len(slices['x']) - 1)]


def native_methods():
    source = ROOT / 'trainer/EIN_trainer.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EINTrainer')
    names = ('validate_epoch', 'test')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict,
                     accuracy_score=accuracy_score, roc_auc_score=roc_auc_score, f1_score=f1_score)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return type('NativeMetricAudit', (), {name: namespace[name] for name in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--threads', type=int, default=4)
    cli = parser.parse_args()
    torch.set_num_threads(cli.threads)
    config = yaml.safe_load(cli.config.read_text())
    assert config['dataset'] == 'DRWeibo' and config['selection_metric'] == 'val_loss'
    assert not config.get('checkpoint_average_metrics')
    model_hash = hashlib.sha256((ROOT / 'model/DIGNN.py').read_bytes()).hexdigest()
    initial = json.loads((ROOT / 'experiments/EIN/DRWeibo/dignn_tuning/initial_state.json').read_text())
    assert model_hash == initial['model_sha256']
    rows = []
    for seed in cli.seeds:
        directory = cli.output / f'seed_{seed}'
        metadata = json.loads((directory / 'validation.json').read_text())
        history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
        minimum = min(history, key=lambda row: row['validation']['loss'])
        chosen = metadata['best']['loss']
        assert len(history) == metadata['completed_epochs']
        assert chosen['epoch'] == minimum['epoch'] and chosen['value'] == minimum['validation']['loss']
        assert (len(history) == config['n_epochs']
                or history[-1]['epoch'] - chosen['epoch'] >= config['patience']), 'Training has not finished'
        assert metadata['model_sha256'] == model_hash
        expected = json.loads((directory / 'test.json').read_text())
        checkpoint = directory / 'best_val_loss.pth'
        assert Path(expected['checkpoint']).resolve() == checkpoint.resolve()
        if 'checkpoint_sha256' in expected:
            assert expected['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            assert expected['selected_epoch'] == chosen['epoch']
        args = SimpleNamespace(**dict(config, seed=seed, device='cpu'))
        model = DIGNN(args.in_feats, args.hidden_dim, args.num_classes, args)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        trainer = native_methods()()
        trainer.model = model
        trainer.logger = SimpleNamespace(info=lambda *args: None)
        trainer._move_to_device = lambda graph: graph
        trainer._accumulate_diagnostics = lambda *args: None
        trainer._average_diagnostics = lambda *args: {}
        trainer._write_val_tensorboard = lambda *args: None
        trainer._write_test_tensorboard = lambda *args: None
        root = cache_root(config, seed)
        trainer.val_loader = DataLoader(complete_graphs(root / 'val/processed/data.pt'), batch_size=args.batch_size)
        trainer.test_loader = DataLoader(complete_graphs(root / 'test/processed/data.pt'), batch_size=args.batch_size)
        validation = trainer.validate_epoch(chosen['epoch'])
        for key in ('loss', 'acc', 'auc', 'f1'):
            assert abs(validation['val_' + key] - chosen['validation'][key]) < 1e-10, (seed, key)
        test = trainer.test()
        predictions = np.load(directory / 'test_predictions.npz')
        y, predicted = predictions['y'], (predictions['probability'] > .5).astype(np.int64)
        tp, tn = int(((y == 1) & (predicted == 1)).sum()), int(((y == 0) & (predicted == 0)).sum())
        fp, fn = int(((y == 0) & (predicted == 1)).sum()), int(((y == 1) & (predicted == 0)).sum())
        manual = dict(acc=(tp + tn) / len(y), auc=.5 * (tp / (tp + fn) + tn / (tn + fp)),
                      f1=2 * tp / (2 * tp + fp + fn))
        for key in ('acc', 'auc', 'f1'):
            assert abs(test[key] - expected['test'][key]) < 1e-12, (seed, key, test, expected)
            assert abs(test[key] - manual[key]) < 1e-12
        row = dict(seed=seed, selected_epoch=chosen['epoch'], validation=validation,
                   test={key: float(value) for key, value in test.items()},
                   confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn),
                   parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(model_source_unchanged=True, model_sha256=model_hash,
                  selection_metric='val_loss', checkpoint_average_metrics=[],
                  native_metrics_match=True, confusion_metrics_match=True, results=rows)
    suffix = '' if sorted(cli.seeds) == list(range(5)) else '_seeds_' + '_'.join(map(str, cli.seeds))
    (cli.output / f'final_audit{suffix}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
