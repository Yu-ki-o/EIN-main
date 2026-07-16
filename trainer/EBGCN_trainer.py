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


class EBGCNTrainer:
    """Training loop for EBGCN's classification and TD/BU KL losses."""

    def __init__(self, datasets, model, optimizer, args, device):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        train_dataset, val_dataset, test_dataset = datasets
        loader_kwargs = self._loader_kwargs()
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, **loader_kwargs)
        self.train_per_epoch = len(self.train_loader)
        self.edge_loss_weight = float(getattr(args, 'ebgcn_edge_loss_weight', 1.0))

        args.log_dir = get_log_dir(args)
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(args.log_dir, 'best_model.pth')

    def _loader_kwargs(self):
        num_workers = max(0, int(getattr(self.args, 'num_workers', 0)))
        kwargs = {'num_workers': num_workers, 'pin_memory': self.device.type == 'cuda'}
        if num_workers > 0:
            kwargs['persistent_workers'] = bool(getattr(self.args, 'persistent_workers', True))
        return kwargs

    def _move_to_device(self, data):
        return data.to(self.device, non_blocking=self.device.type == 'cuda')

    @staticmethod
    def _edge_loss(td_edge_loss, bu_edge_loss, reference):
        losses = [loss for loss in (td_edge_loss, bu_edge_loss) if loss is not None]
        return sum(losses) if losses else reference.sum() * 0.0

    def _selection_metric(self):
        metric = str(getattr(self.args, 'selection_metric', 'val_loss')).strip()
        if metric not in SELECTION_METRIC_MODES:
            raise ValueError('selection_metric must be one of {}, got {}'.format(sorted(SELECTION_METRIC_MODES), metric))
        return metric

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = total_cls_loss = total_edge_loss = 0.0
        for data in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            data = self._move_to_device(data)
            out, td_edge_loss, bu_edge_loss = self.model(data)
            cls_loss = F.nll_loss(out, data.y.view(-1).long())
            edge_loss = self._edge_loss(td_edge_loss, bu_edge_loss, out)
            loss = cls_loss + self.edge_loss_weight * edge_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite EBGCN loss detected at epoch {}.'.format(epoch))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_edge_loss += edge_loss.item()
        average_loss = total_loss / self.train_per_epoch
        self.logger.info('*******Training Epoch {}: Loss {:.6f} | Class {:.6f} | Edge {:.6f}'.format(epoch, average_loss, total_cls_loss / self.train_per_epoch, total_edge_loss / self.train_per_epoch))
        return average_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for data in self.val_loader:
                data = self._move_to_device(data)
                out, _, _ = self.model(data)
                losses.append(F.nll_loss(out, data.y.view(-1).long()).item())
                y_true.extend(data.y.view(-1).tolist())
                y_pred.extend(out.argmax(dim=1).tolist())
        metrics = self._metrics(y_true, y_pred, val_loss=float(np.mean(losses)))
        self.logger.info('*******Val Epoch {}: Loss {:.6f} | Acc {:.4f} | AUC {:.4f} | F1 {:.4f}'.format(epoch, metrics['val_loss'], metrics['val_acc'], metrics['val_auc'], metrics['val_f1']))
        return metrics

    @staticmethod
    def _metrics(y_true, y_pred, val_loss=None):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan
        metrics = {'val_acc': accuracy_score(y_true, y_pred), 'val_auc': auc, 'val_f1': f1_score(y_true, y_pred, zero_division=0)}
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        return metrics

    def test(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data in self.test_loader:
                data = self._move_to_device(data)
                out, _, _ = self.model(data)
                y_true.extend(data.y.view(-1).tolist())
                y_pred.extend(out.argmax(dim=1).tolist())
        metrics = self._metrics(y_true, y_pred)
        result = {'acc': metrics['val_acc'], 'auc': metrics['val_auc'], 'f1': metrics['val_f1']}
        self.logger.info('Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}'.format(result['acc'], result['auc'], result['f1']))
        return result

    def train_process(self):
        start_time = time.time()
        metric = self._selection_metric()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, mode=SELECTION_METRIC_MODES[metric], metric_name=metric)
        for epoch in range(self.args.n_epochs):
            train_loss = self.train_epoch(epoch)
            if train_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            val_metrics = self.validate_epoch(epoch)
            early_stopping(val_metrics[metric], self.model, epoch, self.best_path)
            if early_stopping.early_stop:
                self.logger.info("Validation performance didn't improve for {} epochs. Training stops.".format(self.args.patience))
                break
        self.logger.info('Training finished in {:.2f} min; best {}: {:.4f} at epoch {}.'.format((time.time() - start_time) / 60, metric, early_stopping.best_value, early_stopping.best_epoch))
        best_model_path = self.best_path + '.m'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.test()
