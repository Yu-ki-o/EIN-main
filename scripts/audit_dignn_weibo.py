#!/usr/bin/env python3
"""Audit Weibo CUDA training, val_loss checkpoints and native GPU testing."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import yaml

from audit_dignn_drweibo import complete_graphs, native_methods
from tune_dignn_pheme import ROOT, cache_root
from model.DIGNN import DIGNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    cli = parser.parse_args()
    device = torch.device(cli.device)
    assert device.type == 'cuda' and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = yaml.safe_load(cli.config.read_text())
    assert config['dataset'] == 'Weibo' and config['selection_metric'] == 'val_loss'
    assert torch.device(config['device']).type == 'cuda'
    assert not config.get('checkpoint_average_metrics')
    assert config.get('ema_decay') is None and not config.get('calibrate_threshold')
    initial = json.loads((ROOT / 'experiments/EIN/Weibo/dignn_tuning/initial_state.json').read_text())
    model_hash = hashlib.sha256((ROOT / 'model/DIGNN.py').read_bytes()).hexdigest()
    assert model_hash == initial['model_sha256']
    rows = []
    for seed in cli.seeds:
        directory = cli.output / f'seed_{seed}'
        metadata = json.loads((directory / 'validation.json').read_text())
        stored = yaml.safe_load((directory / 'config.yaml').read_text())
        assert stored == dict(config, seed=seed, device=stored['device'])
        assert torch.device(stored['device']).type == 'cuda'
        assert metadata['config'] == stored
        for name in ('model_device', 'training_device', 'validation_device'):
            assert torch.device(metadata['runtime'][name]).type == 'cuda'
        assert metadata['model_sha256'] == model_hash
        history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
        assert len(history) == metadata['completed_epochs']
        assert [r['epoch'] for r in history] == list(range(len(history)))
        assert all(torch.device(r['training_device']).type == 'cuda' and
                   torch.device(r['validation_device']).type == 'cuda' for r in history)
        minimum = min(history, key=lambda r: r['validation']['loss'])
        chosen = metadata['best']['loss']
        assert chosen['epoch'] == minimum['epoch']
        assert chosen['value'] == minimum['validation']['loss']
        assert len(history) == config['n_epochs'] or history[-1]['epoch'] - chosen['epoch'] >= config['patience']
        expected = json.loads((directory / 'test.json').read_text())
        assert torch.device(expected['runtime']['test_device']).type == 'cuda'
        checkpoint = directory / 'best_val_loss.pth'
        assert Path(expected['checkpoint']).resolve() == checkpoint.resolve()
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert expected['checkpoint_sha256'] == checkpoint_hash
        assert expected['selected_epoch'] == chosen['epoch']
        assert expected['threshold'] == .5
        args = SimpleNamespace(**dict(config, seed=seed, device=str(device)))
        model = DIGNN(args.in_feats, args.hidden_dim, args.num_classes, args).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        trainer = native_methods()()
        trainer.model = model
        trainer.logger = SimpleNamespace(info=lambda *args: None)
        trainer._move_to_device = lambda graph: graph.to(device)
        trainer._accumulate_diagnostics = lambda *args: None
        trainer._average_diagnostics = lambda *args: {}
        trainer._write_val_tensorboard = lambda *args: None
        trainer._write_test_tensorboard = lambda *args: None
        root = cache_root(config, seed)
        trainer.val_loader = DataLoader(complete_graphs(root / 'val/processed/data.pt'), batch_size=args.batch_size)
        trainer.test_loader = DataLoader(complete_graphs(root / 'test/processed/data.pt'), batch_size=args.batch_size)
        validation = trainer.validate_epoch(chosen['epoch'])
        for key in ('loss', 'acc', 'auc', 'f1'):
            assert abs(validation['val_' + key] - chosen['validation'][key]) < 1e-9, (seed, key)
        test = trainer.test()
        archive = np.load(directory / 'test_predictions.npz')
        y = archive['y']
        predicted = (archive['probability'] > .5).astype(np.int64)
        tp = int(((y == 1) & (predicted == 1)).sum())
        tn = int(((y == 0) & (predicted == 0)).sum())
        fp = int(((y == 0) & (predicted == 1)).sum())
        fn = int(((y == 1) & (predicted == 0)).sum())
        manual = dict(acc=(tp + tn) / len(y), auc=.5 * (tp / (tp + fn) + tn / (tn + fp)),
                      f1=2 * tp / (2 * tp + fp + fn))
        for key in ('acc', 'auc', 'f1'):
            assert abs(test[key] - expected['test'][key]) < 1e-12
            assert abs(test[key] - manual[key]) < 1e-12
        rows.append(dict(seed=seed, selected_epoch=chosen['epoch'], completed_epochs=len(history),
                         training_device=stored['device'], audit_device=str(next(model.parameters()).device),
                         validation=validation, test={k: float(v) for k, v in test.items()},
                         confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn), checkpoint_sha256=checkpoint_hash))
        print(json.dumps(rows[-1]), flush=True)
        del model, trainer
        torch.cuda.empty_cache()
    summary = dict(model_source_unchanged=True, model_sha256=model_hash,
                   selection_metric='val_loss', all_training_validation_testing_on_cuda=True,
                   native_metrics_match=True, confusion_metrics_match=True, results=rows)
    suffix = '' if sorted(cli.seeds) == list(range(5)) else '_seeds_' + '_'.join(map(str, cli.seeds))
    (cli.output / f'final_audit{suffix}.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
