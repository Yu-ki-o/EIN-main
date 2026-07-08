import torch.nn.functional as F
import numpy as np
import torch
from collections import defaultdict
from utils.earlystopping import EarlyStopping
from torch_geometric.loader import DataLoader
from utils.dataloader import *
import time
import os
from utils.word2vec import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from utils.logger import (
    get_logger, 
    get_log_dir,
)


SELECTION_METRIC_MODES = {
    'val_loss': 'min',
    'val_acc': 'max',
    'val_auc': 'max',
    'val_f1': 'max',
}


class EINTrainer(object):
    def __init__(self, datasets, model, optimizer, args, device):
        super(EINTrainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.device = device
        self.args = args

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

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
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
        self.tb_writer = None
        self.tb_log_dir = None
        self._last_completed_epoch = 0
        self._init_tensorboard()

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def _use_tensorboard(self):
        for name in ('use_tensorboard', 'enable_tensorboard', 'tensorboard'):
            if hasattr(self.args, name):
                return self._as_bool(getattr(self.args, name))
        return False

    def _init_tensorboard(self):
        if not self._use_tensorboard():
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except (ImportError, ModuleNotFoundError) as exc:
            self.logger.warning(
                'TensorBoard requested but torch.utils.tensorboard is unavailable: {}'.format(exc)
            )
            return

        tb_log_dir = getattr(self.args, 'tensorboard_log_dir', None)
        if tb_log_dir is None or str(tb_log_dir).strip() == '':
            tb_log_dir = os.path.join(self.args.log_dir, 'tensorboard')
        else:
            tb_log_dir = str(tb_log_dir).strip()
            if not os.path.isabs(tb_log_dir):
                tb_log_dir = os.path.join(self.args.log_dir, tb_log_dir)

        os.makedirs(tb_log_dir, exist_ok=True)
        flush_secs = int(getattr(self.args, 'tensorboard_flush_secs', 30))
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir, flush_secs=flush_secs)
        self.tb_log_dir = tb_log_dir
        self.logger.info('TensorBoard log path in: {}'.format(tb_log_dir))
        self._write_tensorboard_config()

    def _write_tensorboard_config(self):
        if self.tb_writer is None:
            return
        config_lines = []
        for key, value in sorted(vars(self.args).items()):
            config_lines.append('{}: {}'.format(key, value))
        self.tb_writer.add_text('config/args', '\n'.join(config_lines), 0)

    def _tb_add_scalar(self, tag, value, step):
        if self.tb_writer is None or value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if np.isnan(value) or np.isinf(value):
            return
        self.tb_writer.add_scalar(tag, value, step)

    def _tb_add_diagnostics(self, prefix, diagnostics, epoch):
        for name, value in diagnostics.items():
            self._tb_add_scalar('ds/{}/{}'.format(prefix, name), value, epoch)

    def _tb_log_learning_rates(self, epoch):
        if self.tb_writer is None:
            return
        for index, param_group in enumerate(self.optimizer.param_groups):
            self._tb_add_scalar(
                'lr/group_{}'.format(index),
                param_group.get('lr'),
                epoch,
            )

    def _tensor_mean(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return value.detach().float().mean().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _model_diagnostics(self):
        diagnostics = {}

        edge_unknown = self._tensor_mean(
            getattr(self.model, '_last_edge_unknown_mass', None)
        )
        if edge_unknown is not None:
            diagnostics['edge_unknown_mass'] = edge_unknown

        global_conflict = self._tensor_mean(
            getattr(self.model, '_last_global_ds_conflict', None)
        )
        if global_conflict is not None:
            diagnostics['global_conflict'] = global_conflict

        global_masses = getattr(self.model, '_last_global_ds_masses', None)
        if torch.is_tensor(global_masses) and global_masses.numel() > 0:
            diagnostics['global_unknown_mass'] = (
                global_masses.detach().float()[..., -1].mean().item()
            )

        branch_masses = getattr(self.model, '_last_global_ds_branch_masses', None)
        if torch.is_tensor(branch_masses) and branch_masses.numel() > 0:
            diagnostics['global_branch_unknown_mass'] = (
                branch_masses.detach().float()[..., -1].mean().item()
            )

        return diagnostics

    def _accumulate_diagnostics(self, sums, counts):
        for name, value in self._model_diagnostics().items():
            sums[name] += value
            counts[name] += 1

    @staticmethod
    def _average_diagnostics(sums, counts):
        averaged = {}
        for name, total in sums.items():
            count = counts.get(name, 0)
            if count > 0:
                averaged[name] = total / count
        return averaged

    def _write_train_tensorboard(self, epoch, metrics, diagnostics):
        self._tb_add_scalar('loss/train_total', metrics.get('train_loss'), epoch)
        self._tb_add_scalar(
            'loss/train_classification',
            metrics.get('train_classification_loss'),
            epoch,
        )
        self._tb_add_scalar(
            'loss/train_physics',
            metrics.get('train_physics_loss'),
            epoch,
        )
        self._tb_add_scalar(
            'loss/train_auxiliary',
            metrics.get('train_auxiliary_loss'),
            epoch,
        )
        self._tb_add_diagnostics('train', diagnostics, epoch)
        self._tb_log_learning_rates(epoch)

    def _write_val_tensorboard(self, epoch, metrics, diagnostics):
        self._tb_add_scalar('loss/val', metrics.get('val_loss'), epoch)
        self._tb_add_scalar('metrics/val_acc', metrics.get('val_acc'), epoch)
        self._tb_add_scalar('metrics/val_auc', metrics.get('val_auc'), epoch)
        self._tb_add_scalar('metrics/val_f1', metrics.get('val_f1'), epoch)
        self._tb_add_diagnostics('val', diagnostics, epoch)

    def _write_test_tensorboard(self, metrics):
        step = getattr(self, '_last_completed_epoch', 0)
        self._tb_add_scalar('metrics/test_acc', metrics.get('acc'), step)
        self._tb_add_scalar('metrics/test_auc', metrics.get('auc'), step)
        self._tb_add_scalar('metrics/test_f1', metrics.get('f1'), step)

    def close_tensorboard(self):
        if self.tb_writer is None:
            return
        self.tb_writer.flush()
        self.tb_writer.close()
        self.tb_writer = None

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
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        train_loss = 0
        train_classification_loss = 0
        train_physics_loss = 0
        train_auxiliary_loss = 0
        diagnostic_sums = defaultdict(float)
        diagnostic_counts = defaultdict(int)
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data = self._move_to_device(data)
            out_labels, U, S, D = self.model(data)
            
            p_loss = self.model.physics_loss(U, S, D, data.user_state)

            if hasattr(self.model, 'classification_loss'):
                classification_loss = self.model.classification_loss(
                    out_labels,
                    data.y,
                )
            else:
                classification_loss = F.nll_loss(out_labels, data.y)
            loss = classification_loss + p_loss
            aux_loss_value = 0.0
            if hasattr(self.model, 'auxiliary_loss'):
                aux_loss = self.model.auxiliary_loss()
                if aux_loss is not None:
                    loss = loss + aux_loss
                    aux_loss_value = aux_loss.item()
            loss.backward()
            train_loss += loss.item()
            train_classification_loss += classification_loss.item()
            train_physics_loss += p_loss.item()
            train_auxiliary_loss += aux_loss_value
            self._accumulate_diagnostics(diagnostic_sums, diagnostic_counts)
            self.optimizer.step()

        train_epoch_loss = train_loss/self.train_per_epoch
        train_metrics = {
            'train_loss': train_epoch_loss,
            'train_classification_loss': (
                train_classification_loss / self.train_per_epoch
            ),
            'train_physics_loss': train_physics_loss / self.train_per_epoch,
            'train_auxiliary_loss': train_auxiliary_loss / self.train_per_epoch,
        }
        train_diagnostics = self._average_diagnostics(
            diagnostic_sums,
            diagnostic_counts,
        )
        self._write_train_tensorboard(epoch, train_metrics, train_diagnostics)

        self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss

    def validate_epoch(self, epoch):

        val_losses = []
        y_true = []
        y_pred = []
        diagnostic_sums = defaultdict(float)
        diagnostic_counts = defaultdict(int)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data = self._move_to_device(data)
                val_out, _, _, _ = self.model(data)
                if hasattr(self.model, 'classification_loss'):
                    val_loss = self.model.classification_loss(
                        val_out,
                        data.y,
                    )
                else:
                    val_loss = F.nll_loss(val_out, data.y)
                val_losses.append(val_loss.item())
                y_true += data.y.tolist()
                y_pred += val_out.max(1).indices.tolist()
                self._accumulate_diagnostics(diagnostic_sums, diagnostic_counts)

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
        val_diagnostics = self._average_diagnostics(
            diagnostic_sums,
            diagnostic_counts,
        )
        self._write_val_tensorboard(epoch, val_metrics, val_diagnostics)
        
        return val_metrics
    
    def test(self):
        # test
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data = self._move_to_device(data)
                test_out, _, _, _ = self.model(data)

                y_true += data.y.tolist()
                y_pred += test_out.max(1).indices.tolist()

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            metrics = {'acc': acc, 'auc': auc, 'f1': f1}
            self.logger.info("Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}".format(acc, auc, f1))
            self._write_test_tensorboard(metrics)
            return metrics
  
        
    def train_process(self):

        try:
            start_time = time.time()

            selection_metric = self.get_selection_metric()
            selection_mode = SELECTION_METRIC_MODES[selection_metric]
            self.logger.info(
                'Checkpoint selection metric: {} ({})'.format(
                    selection_metric, selection_mode
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

                # validation
                val_metrics = self.validate_epoch(epoch)
                
                early_stopping(val_metrics[selection_metric], self.model, epoch, self.best_path)
                self._tb_add_scalar(
                    'selection/current_{}'.format(selection_metric),
                    val_metrics[selection_metric],
                    epoch,
                )
                self._tb_add_scalar(
                    'selection/best_{}'.format(selection_metric),
                    early_stopping.best_value,
                    epoch,
                )
                self._last_completed_epoch = epoch
                
                if early_stopping.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.patience))
                    break
            
            training_time = time.time() - start_time
            self.logger.info("== Training finished.\n"
                        "Total training time: {:.2f} min\t"
                        "best {}: {:.4f}\t"
                        "best epoch: {}\t".format(
                            (training_time / 60), 
                            selection_metric,
                            early_stopping.best_value,
                            early_stopping.best_epoch))

            best_model_path = self.best_path + '.m'
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                self.logger.info("Loaded best checkpoint for testing: {}".format(best_model_path))

            return self.test()
        finally:
            self.close_tensorboard()
