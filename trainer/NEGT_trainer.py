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


class NEGTTrainer(object):
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
            val_dataset,
            batch_size=args.batch_size,
            **loader_kwargs
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            **loader_kwargs
        )
        self.train_per_epoch = len(self.train_loader)
        self.info_loss_weight = float(getattr(args, 'negt_info_loss_weight', 1.0))

        args.log_dir = get_log_dir(args)
        if self._as_bool(getattr(args, 'eval_only', False)):
            early_test_root = str(getattr(args, 'early_test_root', '')).rstrip('/\\')
            cutoff_name = os.path.basename(early_test_root) or 'test'
            args.log_dir = os.path.join(
                args.log_dir,
                'early_detection',
                cutoff_name,
            )
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))
        self.logger.info(
            'Runtime device: {} | pin_memory: {} | non_blocking transfer: {}'.format(
                self.device,
                self.device.type == 'cuda',
                self.device.type == 'cuda',
            )
        )

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

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
        return data.to(
            self.device,
            non_blocking=self.device.type == 'cuda',
        )

    def get_selection_metric(self):
        metric = getattr(self.args, 'selection_metric', 'val_loss')
        if metric is None:
            metric = 'val_loss'
        metric = str(metric).strip()
        if metric not in SELECTION_METRIC_MODES:
            raise ValueError(
                'selection_metric must be one of {}, got {}'.format(
                    sorted(SELECTION_METRIC_MODES), metric
                )
            )
        return metric

    def train_epoch(self, epoch):
        self.model.train()
        self.model.set_epoch(epoch)
        train_loss = 0

        for data in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            data = self._move_to_device(data)
            out, info_loss = self.model(data)
            cls_loss = F.nll_loss(out, data.y.view(-1).long())
            loss = cls_loss + self.info_loss_weight * info_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    'Non-finite NEGT loss detected at epoch {}.'.format(epoch)
                )
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_epoch_loss = train_loss / self.train_per_epoch
        self.logger.info(
            '*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(
                epoch, train_epoch_loss
            )
        )
        return train_epoch_loss

    def validate_epoch(self, epoch):
        val_losses = []
        y_true = []
        y_pred = []
        self.model.eval()
        self.model.set_epoch(epoch)

        with torch.no_grad():
            for data in self.val_loader:
                data = self._move_to_device(data)
                out, _ = self.model(data)
                val_loss = F.nll_loss(out, data.y.view(-1).long())
                val_losses.append(val_loss.item())
                y_true += data.y.view(-1).tolist()
                y_pred += out.max(1).indices.tolist()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(y_true, y_pred)
        try:
            val_auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            val_auc = np.nan
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        val_metrics = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1,
        }
        self.logger.info(
            '*******Val Epoch {}: Loss {:.6f} | Acc {:.4f} | AUC {:.4f} | F1 {:.4f}'.format(
                epoch, val_loss, val_acc, val_auc, val_f1
            )
        )
        return val_metrics

    def test(self):
        y_true = []
        y_pred = []
        self.model.eval()

        with torch.no_grad():
            for data in self.test_loader:
                data = self._move_to_device(data)
                out, _ = self.model(data)
                y_true += data.y.view(-1).tolist()
                y_pred += out.max(1).indices.tolist()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        self.logger.info(
            'Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}'.format(acc, auc, f1)
        )
        return {'acc': acc, 'auc': auc, 'f1': f1}

    def load_evaluation_checkpoint(self):
        checkpoint_path = getattr(self.args, 'checkpoint_path', None)
        if checkpoint_path is None or not str(checkpoint_path).strip():
            raise ValueError('--checkpoint_path is required with --eval_only')
        checkpoint_path = os.path.abspath(
            os.path.expanduser(str(checkpoint_path).strip())
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'Evaluation checkpoint not found: {}'.format(checkpoint_path)
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict', 'model'):
                nested = checkpoint.get(key)
                if isinstance(nested, dict):
                    checkpoint = nested
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(
                'Checkpoint must be a state_dict or contain state_dict/model_state_dict; '
                'got {}'.format(type(checkpoint).__name__)
            )

        model_keys = set(self.model.state_dict())
        legacy_prefixes = {
            'linear1.': 'liner1.',
            'linear2.': 'liner2.',
            'linear3.': 'liner3.',
            'transformer1.': 'Atten_transformer1.',
            'transformer2.': 'Atten_transformer2.',
        }
        should_remap = (
            any(key.startswith(tuple(legacy_prefixes)) for key in checkpoint)
            and any(key.startswith(tuple(legacy_prefixes.values())) for key in model_keys)
        )
        if should_remap:
            remapped = {}
            for key, value in checkpoint.items():
                new_key = key
                for old_prefix, new_prefix in legacy_prefixes.items():
                    if key.startswith(old_prefix):
                        new_key = new_prefix + key[len(old_prefix):]
                        break
                remapped[new_key] = value
            checkpoint = remapped
            self.logger.info('Remapped legacy NEGT checkpoint parameter names.')

        self.model.load_state_dict(checkpoint, strict=True)
        self.logger.info(
            'Loaded evaluation checkpoint: {}'.format(checkpoint_path)
        )

    def train_process(self):
        if self._as_bool(getattr(self.args, 'eval_only', False)):
            self.load_evaluation_checkpoint()
            return self.test()

        start_time = time.time()
        selection_metric = self.get_selection_metric()
        selection_mode = SELECTION_METRIC_MODES[selection_metric]
        self.logger.info(
            'Checkpoint selection metric: {} ({})'.format(
                selection_metric,
                selection_mode,
            )
        )
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            mode=selection_mode,
            metric_name=selection_metric,
        )

        for epoch in range(self.args.n_epochs):
            train_epoch_loss = self.train_epoch(epoch)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_metrics = self.validate_epoch(epoch)
            early_stopping(
                val_metrics[selection_metric],
                self.model,
                epoch,
                self.best_path,
            )
            if early_stopping.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. Training stops.".format(
                        self.args.patience
                    )
                )
                break

        training_time = time.time() - start_time
        self.logger.info(
            '== Training finished.\n'
            'Total training time: {:.2f} min\t'
            'best {}: {:.4f}\t'
            'best epoch: {}\t'.format(
                training_time / 60,
                selection_metric,
                early_stopping.best_value,
                early_stopping.best_epoch,
            )
        )

        best_model_path = self.best_path + '.m'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            self.logger.info(
                'Loaded best checkpoint for testing: {}'.format(best_model_path)
            )

        return self.test()
