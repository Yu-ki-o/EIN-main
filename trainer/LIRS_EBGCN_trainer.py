"""Trainer for the end-to-end LIRS-inspired EBGCN model."""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from utils.earlystopping import EarlyStopping
from utils.logger import get_log_dir, get_logger


SELECTION_METRIC_MODES = {
    'val_loss': 'min',
    'val_acc': 'max',
    'val_auc': 'max',
    'val_f1': 'max',
}


class LIRSEBGCNTrainer:
    LOSS_WEIGHTS = {
        'edge_loss': ('ebgcn_edge_loss_weight', 1.0),
        'shortcut_loss': ('lirs_shortcut_loss_weight', 0.1),
        'independence_loss': ('lirs_hsic_penalty', 0.05),
        'conditional_independence_loss': (
            'lirs_conditional_hsic_penalty',
            0.05,
        ),
        'cluster_loss': ('lirs_cluster_loss_weight', 0.05),
        'infomax_loss': ('lirs_infomax_loss_weight', 0.05),
        'gate_balance_loss': ('lirs_gate_balance_weight', 0.01),
    }

    def __init__(self, datasets, model, optimizer, args, device):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        train_dataset, val_dataset, test_dataset = datasets
        loader_kwargs = self._loader_kwargs()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, **loader_kwargs
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, **loader_kwargs
        )
        self.train_per_epoch = len(self.train_loader)
        self.loss_weights = {
            key: float(getattr(args, config_name, default))
            for key, (config_name, default) in self.LOSS_WEIGHTS.items()
        }

        args.log_dir = get_log_dir(args)
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(args.log_dir, 'best_model.pth')
        self.logger.info('LIRS-EBGCN loss weights: {}'.format(self.loss_weights))

    def _loader_kwargs(self):
        num_workers = max(0, int(getattr(self.args, 'num_workers', 0)))
        kwargs = {
            'num_workers': num_workers,
            'pin_memory': self.device.type == 'cuda',
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = bool(
                getattr(self.args, 'persistent_workers', True)
            )
        return kwargs

    def _move_to_device(self, data):
        return data.to(self.device, non_blocking=self.device.type == 'cuda')

    def _combined_loss(self, out, auxiliary, labels):
        cls_loss = F.nll_loss(out, labels)
        loss = cls_loss
        components = {'class_loss': cls_loss}
        for key, weight in self.loss_weights.items():
            value = auxiliary[key]
            components[key] = value
            if weight != 0:
                loss = loss + weight * value
        return loss, components

    def train_epoch(self, epoch):
        self.model.train()
        totals = {'loss': 0.0}
        for data in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            data = self._move_to_device(data)
            out, auxiliary = self.model(data)
            loss, components = self._combined_loss(
                out, auxiliary, data.y.view(-1).long()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    'Non-finite LIRS-EBGCN loss at epoch {}.'.format(epoch)
                )
            loss.backward()
            gradient_clip = float(
                getattr(self.args, 'lirs_gradient_clip_norm', 5.0)
            )
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip
                )
            self.optimizer.step()
            totals['loss'] += float(loss.item())
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            totals['mean_spurious_gate'] = totals.get(
                'mean_spurious_gate', 0.0
            ) + float(auxiliary['mean_spurious_gate'].item())

        averages = {
            key: value / self.train_per_epoch for key, value in totals.items()
        }
        self.logger.info(
            '*******Training Epoch {}: {}'.format(
                epoch,
                ' | '.join(
                    '{} {:.6f}'.format(key, value)
                    for key, value in averages.items()
                ),
            )
        )
        return averages['loss']

    @staticmethod
    def _metrics(y_true, y_pred, val_loss=None):
        """Compute metrics with the project's existing EIN/EBGCN protocol.

        In particular, AUC is intentionally computed from hard class
        predictions rather than positive-class probabilities to keep reported
        results directly comparable with trainer/EIN_trainer.py and
        trainer/EBGCN_trainer.py.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan
        metrics = {
            'val_acc': accuracy_score(y_true, y_pred),
            'val_auc': auc,
            'val_f1': f1_score(y_true, y_pred, zero_division=0),
        }
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        return metrics

    def _evaluate_loader(self, loader, include_loss):
        self.model.eval()
        losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for data in loader:
                data = self._move_to_device(data)
                out, _ = self.model(data)
                if include_loss:
                    losses.append(
                        F.nll_loss(out, data.y.view(-1).long()).item()
                    )
                y_true.extend(data.y.view(-1).tolist())
                y_pred.extend(out.argmax(dim=1).tolist())
        return self._metrics(
            y_true,
            y_pred,
            val_loss=float(np.mean(losses)) if include_loss else None,
        )

    def validate_epoch(self, epoch):
        metrics = self._evaluate_loader(self.val_loader, include_loss=True)
        self.logger.info(
            '*******Val Epoch {}: Loss {:.6f} | Acc {:.4f} | AUC {:.4f} | F1 {:.4f}'.format(
                epoch,
                metrics['val_loss'],
                metrics['val_acc'],
                metrics['val_auc'],
                metrics['val_f1'],
            )
        )
        return metrics

    def test(self):
        metrics = self._evaluate_loader(self.test_loader, include_loss=False)
        result = {
            'acc': metrics['val_acc'],
            'auc': metrics['val_auc'],
            'f1': metrics['val_f1'],
        }
        self.logger.info(
            'Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}'.format(
                result['acc'], result['auc'], result['f1']
            )
        )
        return result

    def _selection_metric(self):
        metric = str(
            getattr(self.args, 'selection_metric', 'val_loss')
        ).strip()
        if metric not in SELECTION_METRIC_MODES:
            raise ValueError(
                'selection_metric must be one of {}, got {}'.format(
                    sorted(SELECTION_METRIC_MODES), metric
                )
            )
        return metric

    def train_process(self):
        start_time = time.time()
        metric = self._selection_metric()
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            mode=SELECTION_METRIC_MODES[metric],
            metric_name=metric,
        )
        for epoch in range(self.args.n_epochs):
            train_loss = self.train_epoch(epoch)
            if train_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            val_metrics = self.validate_epoch(epoch)
            early_stopping(
                val_metrics[metric], self.model, epoch, self.best_path
            )
            if early_stopping.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. Training stops.".format(
                        self.args.patience
                    )
                )
                break
        self.logger.info(
            'Training finished in {:.2f} min; best {}: {:.4f} at epoch {}.'.format(
                (time.time() - start_time) / 60,
                metric,
                early_stopping.best_value,
                early_stopping.best_epoch,
            )
        )
        best_model_path = self.best_path + '.m'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
        return self.test()
