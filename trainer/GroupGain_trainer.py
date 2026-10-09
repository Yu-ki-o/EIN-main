"""Staged training for GroupGain, with training-only conditional gain targets.

Each split is grouped once on CPU. A separately selected, frozen teacher trains
on complete and randomly masked graphs; inference checkpoints contain only the
student and its grouping settings. This module deliberately does not import the
legacy EIN trainer or dataset module (and their unrelated optional extensions).
"""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model.GroupGain import GroupGain, compute_gain_targets, freeze_teacher
from model.group_gain_grouping import GroupGraph

try:
    from utils.logger import get_log_dir, get_logger
except ImportError:
    # utils.logger imports pandas for an unrelated statistics helper. Keep this
    # trainer usable in a minimal PyTorch/PyG environment without pandas.
    def get_log_dir(args):
        from datetime import datetime
        directory = Path(__file__).resolve().parents[1] / 'experiments' / args.model_name / args.dataset
        group = getattr(args, 'result_group', None)
        if group:
            for part in re.split(r'[\\/]+', str(group)):
                clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', part).strip('._-')
                if clean:
                    directory /= clean
        name = getattr(args, 'result_name', None)
        if name and str(name).strip():
            directory /= str(name).strip().replace('/', '_').replace('\\', '_')
            directory /= 'seed_{}'.format(getattr(args, 'seed', 'run'))
        else:
            directory /= datetime.now().strftime('%Y%m%d-%H%M%S')
        return str(directory)

    def get_logger(root, name=None, debug=True):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        if not debug:
            handler = logging.FileHandler(Path(root) / 'run.log', mode='w')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


SELECTION_METRIC_MODES = {
    'val_loss': 'min', 'val_acc': 'max', 'val_auc': 'max', 'val_f1': 'max',
}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes', 'on'}
    return bool(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _event_id(data):
    for key in ('graph_id', 'event_id', 'tweet_id', 'id'):
        value = getattr(data, key, None)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError('{} must identify one event'.format(key))
            value = value.item()
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError('{} must identify one event'.format(key))
            value = value[0]
        return str(value), key
    return None, None


def _event_fingerprint(data):
    """Compare cached graph content independently of its ID and split."""
    digest = hashlib.sha256()
    identity_fields = {'graph_id', 'event_id', 'tweet_id', 'id', 'split'}
    for key, value in sorted(data.to_dict().items()):
        if key in identity_fields:
            continue
        digest.update(key.encode('utf-8') + b'\0')
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode('utf-8') + b'\0')
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(_json_safe(value), sort_keys=True).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


class GroupGainTrainer:
    """Main-compatible trainer owning teacher, head warm-up, and joint stages."""

    def __init__(self, datasets, model, args, device):
        if not isinstance(model, GroupGain):
            raise TypeError('GroupGainTrainer requires a GroupGain model')
        if len(datasets) != 3:
            raise ValueError('datasets must contain train, validation, and test splits')
        self.args, self.device = args, torch.device(device)
        self.model = model.to(self.device)
        self.teacher = None
        self.history = []
        self.eval_only = _as_bool(getattr(args, 'eval_only', False))
        self.selection_metric = getattr(args, 'selection_metric', 'val_loss')
        if self.selection_metric not in SELECTION_METRIC_MODES:
            raise ValueError('selection_metric must be one of {}'.format(sorted(SELECTION_METRIC_MODES)))
        self.selection_mode = SELECTION_METRIC_MODES[self.selection_metric]
        self.seed = int(getattr(args, 'seed', 0))
        self.mask_generator = torch.Generator().manual_seed(self.seed + 73)
        self.gain_generator = torch.Generator().manual_seed(self.seed + 101)
        self.loader_generator = torch.Generator().manual_seed(self.seed + 131)
        self.budgets = {
            'teacher_max_epochs': self._integer('teacher_max_epochs', 100, minimum=1),
            'teacher_patience': self._integer('teacher_patience', 20, minimum=1),
            'teacher_lr': self._positive('teacher_lr', 5e-4),
            'gain_head_epochs': self._integer('gain_head_epochs', 5, minimum=0),
            'gain_head_lr': self._positive('gain_head_lr', 5e-4),
            'student_max_epochs': self._integer('student_max_epochs', getattr(args, 'n_epochs', 100), minimum=1),
            'student_patience': self._integer('student_patience', getattr(args, 'patience', 20), minimum=1),
            'student_lr': self._positive('student_lr', getattr(args, 'lr', 5e-4)),
        }
        self.mask_probability = float(getattr(args, 'group_gain_teacher_mask_probability', 0.5))
        if not 0 <= self.mask_probability <= 1:
            raise ValueError('group_gain_teacher_mask_probability must be in [0, 1]')
        self.train_batch_size = self._integer('train_batch_size', getattr(args, 'batch_size', 32), minimum=1)
        self.eval_batch_size = self._integer('eval_batch_size', getattr(args, 'batch_size', 32), minimum=1)

        args.log_dir = get_log_dir(args)
        if self.eval_only:
            cutoff = Path(str(getattr(args, 'early_test_root', '')).rstrip('/\\')).name or 'test'
            args.log_dir = str(Path(args.log_dir) / 'early_detection' / cutoff)
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        existing_logger = logging.getLogger(str(self.log_dir))
        for handler in list(existing_logger.handlers):
            handler.close()
            existing_logger.removeHandler(handler)
        self.logger = get_logger(str(self.log_dir), name=str(self.log_dir), debug=_as_bool(getattr(args, 'debug', False)))
        # Always preserve the stage log, including small debug runs.
        if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            file_handler = logging.FileHandler(self.log_dir / 'run.log', mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file_handler)
        self.best_path = str(self.log_dir / 'best_model.pth')
        self.teacher_path = str(self.log_dir / 'best_teacher.pth')
        self.history_path = self.log_dir / 'history.json'
        self.grouping_path = self.log_dir / 'grouping.json'
        self.logger.info('GroupGain log directory: %s | device: %s', self.log_dir, self.device)
        self.logger.info('Stage budgets: %s', self.budgets)
        self.logger.info('Selection: %s; metrics use hard argmax AUC and binary F1 with numeric positive class 1', self.selection_metric)
        if self.eval_only:
            self.load_evaluation_checkpoint()

        self.supervised_gain = (
            self.model.config['variant'] in self.model._GAIN_VARIANTS
            and self.model.config['variant'] != 'attention_control'
            and self.model.config['lambda_gain'] > 0
        )
        self.gain_disabled_reason = None
        self.grouped_datasets = {}
        self.grouping_summary = {}
        self._known_event_splits = {}
        self._known_event_fingerprints = {}
        split_datasets = dict(zip(('train', 'val', 'test'), datasets))
        # Preserve the held-out splits and remove confirmed copies from train.
        # Legacy Weibo manifests can put identical source posts with different
        # filenames in different splits, then write both under the same ID.
        for split in ('val', 'test', 'train'):
            dataset = split_datasets[split]
            # Evaluation does not need to read training/validation data or labels.
            if self.eval_only and split != 'test':
                self.grouped_datasets[split] = []
                continue
            if dataset is None or len(dataset) == 0:
                raise ValueError('The {} split is empty'.format(split))
            self.grouped_datasets[split] = self._prepare_split(dataset, split)
        self.train_loader = self._loader('train', True) if not self.eval_only else None
        self.val_loader = self._loader('val', False) if not self.eval_only else None
        self.test_loader = self._loader('test', False)
        self.train_per_epoch = len(self.train_loader) if self.train_loader is not None else 0
        self._write_json(self.grouping_path, self.grouping_summary)
        if not self.eval_only and self.supervised_gain and not any(
                graph.num_groups > 1 for graph in self.grouped_datasets['train']):
            self.supervised_gain = False
            self.gain_disabled_reason = 'all_training_events_have_K_zero'
            self.logger.info(
                'All training events have K=0 reply groups: skip empty gain terms; '
                'teacher/head epochs executed: 0; student classification budget: %d epochs',
                self.budgets['student_max_epochs'],
            )

    def _integer(self, name, default, minimum):
        value = getattr(self.args, 'group_gain_' + name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('group_gain_{} must be an integer >= {}'.format(name, minimum))
        return value

    def _positive(self, name, default):
        value = float(getattr(self.args, 'group_gain_' + name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError('group_gain_{} must be finite and positive'.format(name))
        return value

    def _dataset_binding(self, dataset, split):
        source = {
            'dataset': str(getattr(self.args, 'dataset', 'unknown')),
            'split': split, 'class': type(dataset).__name__, 'length': len(dataset),
            'root': str(getattr(dataset, 'root', '')),
            'cache_paths': [],
        }
        for path in getattr(dataset, 'processed_paths', ()):
            path = Path(path)
            entry = {'path': str(path.resolve())}
            if path.is_file():
                stat = path.stat()
                entry.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            source['cache_paths'].append(entry)
        indices = getattr(dataset, 'indices', None)
        if callable(indices):
            source['indices'] = list(indices())
        serialized = json.dumps(_json_safe(source), sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()[:20], source

    def _raw_names(self, dataset):
        """Use filename IDs only when preprocessing/index alignment is known."""
        try:
            names = list(dataset.raw_file_names)
        except (AttributeError, OSError, TypeError):
            return None
        if getattr(dataset, 'pre_filter', None) is not None:
            return None
        indices = getattr(dataset, 'indices', None)
        if callable(indices):
            indices = list(indices())
            total = getattr(dataset, 'len', lambda: len(dataset))()
            if len(names) == total and len(indices) == len(dataset):
                return [names[int(index)] for index in indices]
        return names if len(names) == len(dataset) else None

    def _prepare_split(self, dataset, split):
        started = time.perf_counter()
        binding, source = self._dataset_binding(dataset, split)
        names = self._raw_names(dataset)
        prepared, rows, excluded = [], [], []
        seen = set()
        for index in range(len(dataset)):
            data = dataset[index]
            if isinstance(data, GroupGraph):
                raise TypeError('Pass original PyG Data datasets so grouping settings can be validated once')
            data = data.cpu()
            declared_split = getattr(data, 'split', None)
            if declared_split is not None and str(declared_split) != split:
                raise ValueError('Event {} is marked {} but supplied as {}'.format(index, declared_split, split))
            data.split = split
            root_source = 'explicit'
            if getattr(data, 'root_index', None) is None and getattr(data, 'rootindex', None) is None:
                # TreeDataset/ResGCNTreeDataset place the source first in x.
                data.root_index = torch.tensor([0], dtype=torch.long)
                root_source = 'repository_source_node_zero'
            graph_id, id_source = _event_id(data)
            if graph_id is None:
                if names is not None:
                    graph_id, id_source = str(names[index]), 'aligned_raw_filename'
                else:
                    graph_id = '{}:{}:cache:{}:{}'.format(source['dataset'], split, binding, index)
                    id_source = 'dataset_cache_bound_split_index'
                data.graph_id = graph_id
            if graph_id in seen:
                raise ValueError('Duplicate event identifier {!r} in {} split'.format(graph_id, split))
            seen.add(graph_id)
            previous = None
            if id_source != 'dataset_cache_bound_split_index':
                previous = self._known_event_splits.get(graph_id)
                if previous is not None and not (split == 'train' and previous in {'val', 'test'}):
                    raise ValueError('Event identifier {!r} occurs in both {} and {} splits'.format(graph_id, previous, split))
            grouped = self.model.prepare(data)[0].to('cpu')
            grouped.diagnostics.update(split=split, graph_id_source=id_source,
                                       dataset_cache_binding=binding, trainer_root_source=root_source,
                                       dataset_index=index)
            if grouped.y is None or torch.as_tensor(grouped.y).numel() != 1:
                raise ValueError('The {} split requires one label per event for training/metrics'.format(split))
            raw_label = torch.as_tensor(grouped.y).item()
            if isinstance(raw_label, bool) or not math.isfinite(raw_label) or int(raw_label) != raw_label:
                raise ValueError('Event label must be a finite integer class index')
            label = int(raw_label)
            if not 0 <= label < self.model.num_classes:
                raise ValueError('Event label is outside [0, num_classes)')
            if id_source != 'dataset_cache_bound_split_index':
                fingerprint = _event_fingerprint(data)
                if previous is not None:
                    if fingerprint != self._known_event_fingerprints[graph_id]:
                        raise ValueError(
                            'Event identifier {!r} occurs in both {} and {} splits '
                            'with different graph content or labels'.format(graph_id, previous, split)
                        )
                    excluded.append(dict(graph_id=graph_id, dataset_index=index,
                                         retained_split=previous, graph_id_source=id_source))
                    self.logger.warning(
                        'Exclude duplicate training event %r at dataset index %d: '
                        'identical graph and label retained in %s split', graph_id, index, previous,
                    )
                    continue
                self._known_event_splits[graph_id] = split
                self._known_event_fingerprints[graph_id] = fingerprint
            prepared.append(grouped)
            rows.append(dict(graph_id=grouped.graph_id, **grouped.diagnostics))
        if not prepared:
            raise ValueError('The {} split is empty after excluding held-out duplicates'.format(split))
        original_nodes = sum(graph.num_nodes for graph in prepared)
        group_count = sum(graph.num_groups for graph in prepared)
        summary = {
            'events': len(prepared), 'input_events': len(dataset),
            'excluded_overlap_events': excluded, 'original_nodes': original_nodes,
            'group_nodes': group_count, 'compression_ratio': 1 - group_count / original_nodes,
            'retained_fraction': group_count / original_nodes,
            'preprocessing_seconds': time.perf_counter() - started,
            'dataset_cache_binding': binding, 'cache_identity': source,
            'event_diagnostics': rows,
        }
        self.grouping_summary[split] = summary
        self.logger.info('Grouping %s: events=%d nodes=%d groups=%d compression=%.4f seconds=%.3f',
                         split, len(prepared), original_nodes, group_count,
                         summary['compression_ratio'], summary['preprocessing_seconds'])
        return prepared

    def _loader(self, split, shuffle):
        return DataLoader(
            self.grouped_datasets[split],
            batch_size=self.train_batch_size if shuffle else self.eval_batch_size,
            shuffle=shuffle, collate_fn=list, num_workers=0,
            generator=self.loader_generator if shuffle else None,
        )

    @staticmethod
    def _metrics(labels, predictions):
        # Preserve the existing repository's metric convention for comparison.
        auc = float('nan') if len(set(labels)) < 2 else float(roc_auc_score(labels, predictions))
        return {
            'acc': float(accuracy_score(labels, predictions)), 'auc': auc,
            'f1': float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        }

    @torch.no_grad()
    def evaluate(self, model, loader, use_gain=None):
        started = time.perf_counter()
        model.eval()
        total_loss, total_events = 0.0, 0
        labels, predictions = [], []
        for graphs in loader:
            logits = model.forward_graphs(graphs, use_gain=use_gain)
            target = model._labels(graphs)
            count = len(graphs)
            total_loss += float(model.classification_loss(logits, target)) * count
            total_events += count
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
        if not total_events:
            raise ValueError('Cannot evaluate an empty split')
        return dict(loss=total_loss / total_events, **self._metrics(labels, predictions),
                    seconds=time.perf_counter() - started)

    def _gain_targets(self, graphs):
        return compute_gain_targets(
            self.teacher, graphs,
            candidates_per_graph=self.model.config['candidates_per_graph'],
            contexts_per_candidate=self.model.config['contexts_per_candidate'],
            full_context_probability=(1.0 if self.model.config['variant'] == 'leave_one_out_only'
                                      else self.model.config['full_context_probability']),
            correct_only=self.model.config['correct_only'],
            generator=self.gain_generator,
            micro_batch_size=self.model.config['micro_batch_size'], split='train',
        )

    def train_epoch(self, stage, model, optimizer):
        started = time.perf_counter()
        model.train()
        totals = dict(loss=0.0, classification_loss=0.0, gain_loss=0.0)
        events = batches = samples = positive = negative = zero = 0
        eligible = gain_events = teacher_forwards = 0
        gain_sum = gate_sum = 0.0
        gain_square_sum = gate_square_sum = gain_mae_sum = 0.0
        gain_min = gate_min = math.inf
        gain_max = gate_max = -math.inf
        gate_count = 0
        target_seconds = 0.0
        for graphs in self.train_loader:
            optimizer.zero_grad(set_to_none=True)
            targets = None
            if stage != 'teacher' and self.supervised_gain:
                target_start = time.perf_counter()
                targets = self._gain_targets(graphs)
                target_seconds += time.perf_counter() - target_start
                values = targets.values
                samples += values.numel()
                positive += int((values > 0).sum())
                negative += int((values < 0).sum())
                zero += int((values == 0).sum())
                gain_sum += float(values.sum())
                gain_square_sum += float(values.square().sum())
                if values.numel():
                    gain_min = min(gain_min, float(values.min()))
                    gain_max = max(gain_max, float(values.max()))
                eligible += targets.diagnostics['eligible_events']
                gain_events += targets.diagnostics['events']
                teacher_forwards += targets.diagnostics['teacher_forwards']
            if stage == 'teacher':
                loss = model.teacher_training_loss(graphs, self.mask_probability, self.mask_generator)
                classification, gain = loss, loss.new_zeros(())
            elif stage == 'gain_head':
                loss = model.gain_loss(graphs, targets)
                classification, gain = loss.new_zeros(()), loss
            else:
                loss = model.compute_loss(graphs, gain_targets=targets,
                                          lambda_gain=None if self.supervised_gain else 0.0)
                diagnostics = model.get_diagnostics()
                classification = diagnostics['classification_loss']
                gain = diagnostics['gain_loss']
                gates = diagnostics.get('gates')
                roots = diagnostics.get('root_mask')
                if gates is not None:
                    reply_gates = gates[~roots]
                    gate_sum += float(reply_gates.sum())
                    gate_square_sum += float(reply_gates.square().sum())
                    gate_count += reply_gates.numel()
                    if reply_gates.numel():
                        gate_min = min(gate_min, float(reply_gates.min()))
                        gate_max = max(gate_max, float(reply_gates.max()))
            if targets is not None and targets.values.numel():
                gain_mae_sum += float(model.get_diagnostics()['gain_mae']) * targets.values.numel()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite {} training loss'.format(stage))
            # Empty eligible batches contribute no Huber gradient during warm-up.
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            count = len(graphs)
            events += count
            batches += 1
            totals['loss'] += float(loss.detach()) * count
            totals['classification_loss'] += float(classification.detach()) * count
            totals['gain_loss'] += float(gain.detach()) * count
        if stage != 'teacher' and self.supervised_gain and samples == 0:
            raise RuntimeError(
                'No eligible training gain targets were produced. Check reply groups and teacher '
                'accuracy; train the teacher longer or explicitly set group_gain_correct_only=false.'
            )
        return dict(
            **{key: value / events for key, value in totals.items()}, events=events, batches=batches,
            seconds=time.perf_counter() - started, gain_target_seconds=target_seconds,
            gain_samples=samples, gain_positive=positive, gain_negative=negative, gain_zero=zero,
            gain_positive_fraction=positive / samples if samples else None,
            gain_negative_fraction=negative / samples if samples else None,
            mean_gain_target=gain_sum / samples if samples else None,
            gain_target_min=gain_min if samples else None,
            gain_target_max=gain_max if samples else None,
            gain_target_std=math.sqrt(max(0.0, gain_square_sum / samples - (gain_sum / samples) ** 2)) if samples else None,
            gain_mae=gain_mae_sum / samples if samples else None,
            eligible_fraction=eligible / gain_events if gain_events else None,
            mean_reply_gate=gate_sum / gate_count if gate_count else None,
            reply_gate_min=gate_min if gate_count else None,
            reply_gate_max=gate_max if gate_count else None,
            reply_gate_std=math.sqrt(max(0.0, gate_square_sum / gate_count - (gate_sum / gate_count) ** 2)) if gate_count else None,
            teacher_forwards=teacher_forwards,
        )

    def _checkpoint_metadata(self, stage, epoch, score):
        return {
            'stage': stage, 'selected_epoch': epoch, 'selection_metric': self.selection_metric,
            'selection_value': float(score), 'dataset': str(getattr(self.args, 'dataset', 'unknown')),
            'seed': self.seed, 'training_budgets': dict(self.budgets),
            'teacher_mask_probability': self.mask_probability,
            'gain_supervision_active': self.supervised_gain,
            'gain_disabled_reason': self.gain_disabled_reason,
            'teacher_forward_use_gain': False if stage == 'teacher' else None,
            'root_convention': 'existing explicit root; otherwise repository source at original node 0',
            'label_mapping': getattr(self.args, 'label_mapping', None),
            'metric_convention': 'hard argmax AUC; binary F1 numeric positive class 1; one-class AUC is NaN',
            'grouping_cache_bindings': {split: summary['dataset_cache_binding']
                                       for split, summary in self.grouping_summary.items()},
        }

    def _record(self, stage, epoch, train, validation=None, improved=None):
        row = dict(stage=stage, epoch=epoch, train=train)
        if validation is not None:
            row['validation'] = validation
        if improved is not None:
            row['selected'] = improved
        self.history.append(row)
        self._write_json(self.history_path, self.history)
        self.logger.info('Stage %s epoch %s: %s', stage, epoch, json.dumps(_json_safe(row), sort_keys=True))

    @staticmethod
    def _write_json(path, value):
        with Path(path).open('w', encoding='utf-8') as handle:
            json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)

    def _selected_stage(self, stage, model, optimizer, max_epochs, patience, path):
        best = math.inf if self.selection_mode == 'min' else -math.inf
        stale = 0
        for epoch in range(1, max_epochs + 1):
            train = self.train_epoch(stage, model, optimizer)
            validation = self.evaluate(model, self.val_loader, use_gain=False if stage == 'teacher' else None)
            score = validation[self.selection_metric.removeprefix('val_')]
            if not math.isfinite(score):
                raise RuntimeError('{} is undefined on validation; select val_loss or a defined metric'.format(self.selection_metric))
            improved = score < best if self.selection_mode == 'min' else score > best
            if improved:
                best, stale = score, 0
                model.save_checkpoint(path, metadata=self._checkpoint_metadata(stage, epoch, score))
            else:
                stale += 1
            self._record(stage, epoch, train, validation, improved)
            if stale >= patience:
                self.logger.info('Early stopping %s after epoch %d, patience=%d', stage, epoch, patience)
                break
        restored = GroupGain.load_checkpoint(path, map_location=self.device)
        model.load_state_dict(restored.state_dict())
        model.checkpoint_metadata = restored.checkpoint_metadata
        self.logger.info('Selected %s checkpoint: %s', stage, path)

    def load_evaluation_checkpoint(self):
        path = getattr(self.args, 'checkpoint_path', None)
        if path is None or not str(path).strip():
            raise ValueError('eval_only requires checkpoint_path for a standalone GroupGain student')
        path = Path(str(path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Evaluation checkpoint not found: {}'.format(path))
        self.model = GroupGain.load_checkpoint(path, map_location=self.device)
        metadata = self.model.checkpoint_metadata
        if metadata.get('stage') == 'teacher':
            raise ValueError('Evaluation requires a student checkpoint; a teacher checkpoint was supplied')
        dataset = metadata.get('dataset')
        if dataset is not None and dataset != str(getattr(self.args, 'dataset', 'unknown')):
            raise ValueError('Checkpoint dataset {} does not match requested {}'.format(dataset, self.args.dataset))
        seed = metadata.get('seed')
        if seed is not None and int(seed) != self.seed:
            raise ValueError('Checkpoint seed {} does not match requested {}'.format(seed, self.seed))
        self.logger.info('Loaded standalone student checkpoint before grouping: %s', path)

    def test(self):
        result = self.evaluate(self.model, self.test_loader)
        metrics = {key: result[key] for key in ('acc', 'auc', 'f1')}
        self.history.append(dict(stage='test', metrics=metrics, seconds=result['seconds']))
        self._write_json(self.history_path, self.history)
        self.logger.info('Test Acc: %.4f | AUC: %.4f | F1: %.4f | seconds: %.3f',
                         metrics['acc'], metrics['auc'], metrics['f1'], result['seconds'])
        return metrics

    def train_process(self):
        try:
            if self.eval_only:
                return self.test()
            weight_decay = float(self.model.config['weight_decay'])
            if self.supervised_gain:
                self.teacher = GroupGain(
                    self.model.in_feats, self.model.hidden_dim, self.model.num_classes,
                    args=SimpleNamespace(max_hop=self.model.max_hop), **dict(self.model.config),
                ).to(self.device)
                teacher_optimizer = torch.optim.Adam(
                    self.teacher.parameters(), lr=self.budgets['teacher_lr'], weight_decay=weight_decay,
                )
                self._selected_stage(
                    'teacher', self.teacher, teacher_optimizer,
                    self.budgets['teacher_max_epochs'], self.budgets['teacher_patience'], self.teacher_path,
                )
                freeze_teacher(self.teacher)
                self.model.initialize_from_teacher(self.teacher)
                self.model.requires_grad_(False)
                heads = self.model.gain_head_parameters()
                for parameter in heads:
                    parameter.requires_grad_(True)
                head_optimizer = torch.optim.Adam(heads, lr=self.budgets['gain_head_lr'], weight_decay=weight_decay)
                for epoch in range(1, self.budgets['gain_head_epochs'] + 1):
                    train = self.train_epoch('gain_head', self.model, head_optimizer)
                    self._record('gain_head', epoch, train)
                self.model.requires_grad_(True)
            else:
                self.logger.info(
                    'Variant %s with lambda_gain=%s uses direct classification training; '
                    'teacher and gain-head epochs executed: 0; student budget: %d epochs',
                    self.model.config['variant'], self.model.config['lambda_gain'], self.budgets['student_max_epochs'],
                )
            student_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.budgets['student_lr'], weight_decay=weight_decay,
            )
            self._selected_stage(
                'student', self.model, student_optimizer,
                self.budgets['student_max_epochs'], self.budgets['student_patience'], self.best_path,
            )
            return self.test()
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError(
                    'GroupGain ran out of device memory. Reduce group_gain_train_batch_size, '
                    'group_gain_eval_batch_size and group_gain_micro_batch_size (teacher target inference).'
                ) from error
            raise
        finally:
            for handler in list(self.logger.handlers):
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
