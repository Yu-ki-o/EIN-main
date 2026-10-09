#!/usr/bin/env python3
"""Train unchanged DIGNN directly from existing Pheme/DRWeibo/Weibo PyG caches.

Validation is the default; test caches are read only with --test. This avoids
loading unrelated encoders/backbones when already processed graphs exist.
ACC, hard-label AUC and positive-class F1 match EINTrainer; probability AUC
is separately named. Configs and per-epoch validation records are preserved.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.DIGNN import DIGNN
from utils.checkpoint_average import average_checkpoints, DEFAULT_TARGETS

TARGETS = DEFAULT_TARGETS


def cache_root(config, seed):
    if (config['dataset'] not in ('Pheme', 'DRWeibo', 'Weibo') or config['base_model'] != 'DIGNN'
            or config['word_embedding'] != 'word2vec'
            or config.get('experiment_mode', 'id') != 'id'):
        raise ValueError('This runner requires in-domain Pheme/DRWeibo/Weibo DIGNN Word2Vec caches')
    name = (f"mode-id__graph-resgcn-tree__emb-word2vec__lang-{config['language']}"
            f"__hop-{config['max_hop']}__centrality-PageRank"
            f"__undir-{config['undirected']}__tok-{config['tokenize_mode']}"
            f"__vec-{config['vector_size']}")
    return (ROOT / 'dataset' / config['dataset'] / 'dataset_cache' / name
            / f"split_{config['split']}_k{config['k']}" / f'seed_{seed}')


def load_graphs(path):
    cached, slices = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    graphs = []
    for index in range(len(slices['x']) - 1):
        full = separate(cached.__class__, cached, index, slices, decrement=False)
        graph = Data(x=full.x, edge_index=full.edge_index, y=full.y)
        for name in ('directed_edge_index', 'root_index', 'rootindex'):
            value = getattr(full, name, None)
            if value is not None:
                graph[name] = value
        graphs.append(graph)
    return graphs


def metrics(y, probability, threshold=0.5):
    predicted = (probability > threshold).astype(np.int64)
    return dict(acc=float(accuracy_score(y, predicted)),
                auc=float(roc_auc_score(y, predicted)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                probability_auc=float(roc_auc_score(y, probability)))


def validation_threshold(y, probability):
    """Fixed, predeclared search; labels from the validation split only."""
    candidates = np.linspace(.2, .8, 121)
    return float(max(candidates, key=lambda t: min(
        metrics(y, probability, t)[key] / target for key, target in TARGETS.items())))


def calibrate_saved(output_dir, seeds):
    """Keep raw results intact while evaluating validation-selected thresholds."""
    destination = output_dir / 'calibration'
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        directory = output_dir / f'seed_{seed}'
        metadata = json.loads((directory / 'validation.json').read_text())
        selection = metadata['config']['selection_metric'].removeprefix('val_')
        validation = np.load(directory / f'val_{selection}.npz')
        test = np.load(directory / 'test_predictions.npz')
        # Recalculate the validation values to reject stale prediction archives.
        original = metrics(validation['y'], validation['probability'])
        for key in TARGETS:
            if abs(original[key] - metadata['best'][selection]['validation'][key]) > 1e-12:
                raise ValueError(f'Stale validation archive: {directory}, {key}')
        threshold = validation_threshold(validation['y'], validation['probability'])
        result = dict(seed=seed, threshold=threshold,
                      threshold_rule=f'max validation min(metric/target): {TARGETS}',
                      threshold_grid=dict(min=.2, max=.8, count=121),
                      validation=metrics(validation['y'], validation['probability'], threshold),
                      test=metrics(test['y'], test['probability'], threshold),
                      checkpoint=str(directory / f'best_val_{selection}.pth'),
                      model_sha256=metadata['model_sha256'])
        results.append(result)
        write_json(destination / f'seed_{seed}.json', result)
        print(json.dumps(result), flush=True)
    summary = dict(seeds=seeds, results=results)
    for split in ('validation', 'test'):
        summary[split] = {key: dict(mean=float(np.mean([r[split][key] for r in results])),
                                   std=float(np.std([r[split][key] for r in results])))
                          for key in ('acc', 'auc', 'f1', 'probability_auc')}
    summary['targets'] = TARGETS
    summary['meets_targets'] = (sorted(seeds) == list(range(5)) and all(
        summary['test'][key]['mean'] >= value for key, value in TARGETS.items()))
    suffix = '' if sorted(seeds) == list(range(5)) else '_seeds_' + '_'.join(map(str, seeds))
    write_json(destination / f'test_summary{suffix}.json', summary)
    print(json.dumps(summary), flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, losses = [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)[0]
        losses.append(float(model.classification_loss(output, batch.y)))
        labels.append(batch.y.cpu().numpy().reshape(-1))
        probabilities.append(output[:, 1].exp().cpu().numpy())
    y, probability = np.concatenate(labels), np.concatenate(probabilities)
    values = metrics(y, probability)
    values['loss'] = float(np.mean(losses))  # Same batch averaging as EINTrainer.
    # Joint validation score: weakest fraction of the three requested targets.
    values['target_score'] = min(values[key] / target for key, target in TARGETS.items())
    return values, y, probability


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run(config, seed, output_dir, device, test=False, evaluate_only=False):
    start = time.monotonic()
    directory = output_dir / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(config, seed=seed, device=str(device))
    selection = config.get('selection_metric', 'val_loss').removeprefix('val_')
    if selection not in ('loss', 'acc', 'auc', 'f1', 'target_score'):
        raise ValueError(f'Unsupported selection metric: {selection}')
    model_hash = hashlib.sha256((ROOT / 'model/DIGNN.py').read_bytes()).hexdigest()
    root = cache_root(config, seed)
    val = load_graphs(root / 'val/processed/data.pt')
    train = [] if evaluate_only else load_graphs(root / 'train/processed/data.pt')
    args = SimpleNamespace(**config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    model = DIGNN(args.in_feats, args.hidden_dim, args.num_classes, args).to(device)
    actual_device = next(model.parameters()).device
    runtime = dict(torch_version=torch.__version__, model_device=str(actual_device),
                   training_device=str(actual_device), validation_device=str(actual_device),
                   test_device=str(actual_device) if test else None, precision='float32',
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if actual_device.type == 'cuda':
        runtime.update(gpu_name=torch.cuda.get_device_name(actual_device),
                       cuda_version=torch.version.cuda,
                       matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=bool(train))
    val_loader = DataLoader(val, batch_size=args.batch_size)
    optimizer = model.init_optimizer(args)
    ema_decay = config.get('ema_decay')
    if ema_decay is not None and not 0 <= float(ema_decay) < 1:
        raise ValueError('ema_decay must be in [0, 1)')
    averaged_model = None if ema_decay is None else copy.deepcopy(model).eval().requires_grad_(False)
    updates = 0
    class_weights = config.get('class_weights')
    weights = None if class_weights is None else torch.tensor(class_weights, device=device)
    scheduler = None
    if config.get('lr_scheduler') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.get('lr_factor', .5),
            patience=config.get('lr_patience', 5), min_lr=config.get('min_lr', 1e-5))
    best = {}
    history_path = directory / 'history.jsonl'
    config_path = directory / 'config.yaml'
    checkpoint = directory / f'best_val_{selection}.pth'
    if not evaluate_only:
        if history_path.exists():
            raise FileExistsError(f'Use a new output directory; already exists: {history_path}')
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(json.dumps(dict(seed=seed, train_events=len(train), val_events=len(val),
                              train_nodes=sum(g.num_nodes for g in train),
                              class_counts=np.bincount([int(g.y.item()) for g in train]).tolist(),
                              model_sha256=model_hash, output=str(directory))), flush=True)
        stale = 0
        with history_path.open('w') as history:
            for epoch in range(args.n_epochs):
                epoch_start = time.monotonic()
                model.train()
                total, auxiliary_total = 0., 0.
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch)[0]
                    classification = (model.classification_loss(output, batch.y) if weights is None
                                      else F.nll_loss(output, batch.y.reshape(-1).long(), weight=weights))
                    auxiliary = model.auxiliary_loss()
                    loss = classification + auxiliary
                    if not torch.isfinite(loss):
                        raise ValueError(f'Nonfinite training loss at seed {seed}, epoch {epoch}')
                    loss.backward()
                    clip = config.get('gradient_clip', 0.)
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    if averaged_model is not None:
                        decay = 0. if updates == 0 else float(ema_decay)
                        with torch.no_grad():
                            for average, current in zip(averaged_model.parameters(), model.parameters()):
                                average.mul_(decay).add_(current.detach(), alpha=1 - decay)
                            for average, current in zip(averaged_model.buffers(), model.buffers()):
                                average.copy_(current)
                        updates += 1
                    total += float(loss.detach())
                    auxiliary_total += float(auxiliary.detach())
                evaluation_model = model if averaged_model is None else averaged_model
                values, y, probability = evaluate(evaluation_model, val_loader, device)
                improved_selection = False
                for key in ('loss', 'acc', 'auc', 'f1', 'target_score'):
                    previous = best.get(key)
                    improved = previous is None or (values[key] < previous['value'] if key == 'loss'
                                                   else values[key] > previous['value'])
                    if improved:
                        best[key] = dict(epoch=epoch, value=values[key], validation=values)
                        torch.save(evaluation_model.state_dict(), directory / f'best_val_{key}.pth')
                        np.savez(directory / f'val_{key}.npz', y=y, probability=probability)
                        improved_selection |= key == selection
                stale = 0 if improved_selection else stale + 1
                if scheduler is not None:
                    scheduler.step(values['loss'])
                record = dict(epoch=epoch, seed=seed, train_loss=total / len(train_loader),
                              auxiliary_loss=auxiliary_total / len(train_loader), validation=values,
                              training_device=str(actual_device), validation_device=str(actual_device),
                              lr=optimizer.param_groups[0]['lr'],
                              seconds=time.monotonic() - epoch_start)
                history.write(json.dumps(record) + '\n')
                history.flush()
                print(json.dumps(record), flush=True)
                write_json(directory / 'validation.json', dict(
                    seed=seed, config=config, model_sha256=model_hash, cache=str(root),
                    runtime=runtime,
                    best=best, completed_epochs=epoch + 1, seconds=time.monotonic() - start))
                if stale >= args.patience:
                    break
    else:
        stored = yaml.safe_load(config_path.read_text())
        # A saved model can be evaluated on another device without retraining.
        postprocessing_keys = {'calibrate_threshold', 'checkpoint_average_metrics', 'device'}
        training_stored = {key: value for key, value in stored.items() if key not in postprocessing_keys}
        training_config = {key: value for key, value in config.items() if key not in postprocessing_keys}
        if training_stored != training_config:
            raise ValueError('Evaluation config must match the saved training config')
        metadata = json.loads((directory / 'validation.json').read_text())
        if metadata['model_sha256'] != model_hash:
            raise ValueError('DIGNN.py changed since training')
        runtime['training_device'] = metadata.get('runtime', {}).get('training_device', stored['device'])
    artifact_directory = directory
    average_sources = None
    selectors = config.get('checkpoint_average_metrics')
    if selectors:
        metadata = json.loads((directory / 'validation.json').read_text())
        state, average_sources = average_checkpoints(directory, metadata, selectors, device)
        artifact_directory = directory / 'averages' / '_'.join(selectors)
        artifact_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = artifact_directory / 'best_model.pth'
        torch.save(state, checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    values, y, probability = evaluate(model, val_loader, device)
    metadata = json.loads((directory / 'validation.json').read_text())
    result = dict(seed=seed, validation=values, checkpoint=str(checkpoint), model_sha256=model_hash,
                  runtime=runtime,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  selected_epoch=None if average_sources is not None else metadata['best'][selection]['epoch'])
    if average_sources is not None:
        result['averaged_sources'] = average_sources
        np.savez(artifact_directory / 'validation_predictions.npz', y=y, probability=probability)
        write_json(artifact_directory / 'validation.json', result)
    if test:
        test_graphs = load_graphs(root / 'test/processed/data.pt')
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
        test_values, test_y, test_probability = evaluate(model, test_loader, device)
        # Optional threshold is selected exclusively on this seed's validation set.
        threshold = .5
        if config.get('calibrate_threshold', False):
            threshold = validation_threshold(y, probability)
            result['test_uncalibrated'] = dict(test_values)
            test_values.update(metrics(test_y, test_probability, threshold))
            test_values['target_score'] = min(test_values[key] / target
                                            for key, target in TARGETS.items())
            result['validation_uncalibrated'] = result['validation']
            result['validation'] = dict(values, **metrics(y, probability, threshold))
        result.update(test=test_values, test_events=len(test_graphs), threshold=threshold)
        np.savez(artifact_directory / 'test_predictions.npz', y=test_y, probability=test_probability,
                 threshold=np.array(threshold))
        write_json(artifact_directory / 'test.json', result)
    print(json.dumps(result), flush=True)
    if hashlib.sha256((ROOT / 'model/DIGNN.py').read_bytes()).hexdigest() != model_hash:
        raise ValueError('DIGNN.py changed during training')
    return result


def main(default_config='configs/EIN/Pheme_DIGNN_word2vec.yaml', strict_val_loss=False,
         default_device='cpu', require_cuda=False):
    global TARGETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--calibrate-saved', action='store_true',
                        help='Calibrate existing prediction archives using validation labels only')
    parser.add_argument('--average-checkpoints', nargs='+',
                        choices=['loss', 'acc', 'auc', 'f1', 'target_score'],
                        help='Average unique validation-selected epochs within each seed')
    cli = parser.parse_args()
    if len(set(cli.seeds)) != len(cli.seeds) or not set(cli.seeds) <= set(range(5)):
        parser.error('Use unique seeds from the fixed set 0, 1, 2, 3, 4')
    torch.set_num_threads(cli.threads)
    torch.set_num_interop_threads(1)
    config = yaml.safe_load(Path(cli.config).read_text())
    if require_cuda and (torch.device(cli.device).type != 'cuda' or not torch.cuda.is_available()):
        parser.error('This experiment requires CUDA for both training and testing')
    if require_cuda and config['dataset'] != 'Weibo':
        parser.error('This entry point requires the Weibo dataset')
    if require_cuda and cli.evaluate_only:
        for seed in cli.seeds:
            stored = yaml.safe_load((cli.output / f'seed_{seed}' / 'config.yaml').read_text())
            if torch.device(stored['device']).type != 'cuda':
                parser.error('Weibo checkpoints must have been trained on CUDA')
    if cli.average_checkpoints:
        config['checkpoint_average_metrics'] = cli.average_checkpoints
    if strict_val_loss and (cli.calibrate_saved or config.get('selection_metric') != 'val_loss'
                            or config.get('checkpoint_average_metrics')
                            or config.get('ema_decay') is not None
                            or config.get('calibrate_threshold', False)):
        parser.error('Strict tuning uses only the minimum-val_loss checkpoint, without averaging or calibration')
    TARGETS = {'DRWeibo': dict(acc=.895, auc=.895, f1=.895),
               'Weibo': dict(acc=.96, auc=.96, f1=.96)}.get(config['dataset'], DEFAULT_TARGETS)
    if cli.calibrate_saved:
        if cli.test or cli.evaluate_only:
            parser.error('--calibrate-saved is a separate archive evaluation mode')
        calibrate_saved(cli.output, cli.seeds)
        return
    results = [run(config, seed, cli.output, torch.device(cli.device), cli.test, cli.evaluate_only)
               for seed in cli.seeds]
    summary = dict(seeds=cli.seeds, results=results)
    for split in ('validation', 'test'):
        if all(split in item for item in results):
            summary[split] = {metric: dict(mean=float(np.mean([item[split][metric] for item in results])),
                                          std=float(np.std([item[split][metric] for item in results])))
                              for metric in ('acc', 'auc', 'f1', 'probability_auc')}
    if cli.test and sorted(cli.seeds) == list(range(5)):
        summary['targets'] = TARGETS
        summary['meets_targets'] = all(summary['test'][key]['mean'] >= value
                                       for key, value in summary['targets'].items())
    prefix = 'test' if cli.test else 'validation'
    suffix = '' if sorted(cli.seeds) == list(range(5)) else '_seeds_' + '_'.join(map(str, cli.seeds))
    summary_directory = cli.output
    if config.get('checkpoint_average_metrics'):
        summary_directory = cli.output / 'averages' / '_'.join(config['checkpoint_average_metrics'])
        summary_directory.mkdir(parents=True, exist_ok=True)
    write_json(summary_directory / f'{prefix}_summary{suffix}.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
