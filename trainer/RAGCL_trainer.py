import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from utils.earlystopping import EarlyStopping
from utils.logger import get_log_dir, get_logger
from utils.ragcl_augmentation import augment


SELECTION_METRIC_MODES = {
    'val_loss': 'min',
    'val_acc': 'max',
    'val_auc': 'max',
    'val_f1': 'max',
}


class RAGCLTrainer(object):
    def __init__(self, datasets, model, optimizer, args, device):
        super(RAGCLTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args

        train_dataset, val_dataset, test_dataset = datasets
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        self.train_per_epoch = len(self.train_loader)

        args.log_dir = get_log_dir(args)
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

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
        train_loss = 0
        use_unsup_loss = getattr(self.args, 'use_unsup_loss', False)
        lamda = getattr(self.args, 'ragcl_lamda', getattr(self.args, 'lamda', 0.0))
        aug1 = getattr(self.args, 'aug1', 'DropEdge,mean,0.2,0.7').split('||')
        aug2 = getattr(self.args, 'aug2', 'NodeDrop,0.2,0.7').split('||')

        for data in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            out = self.model(data)
            sup_loss = F.nll_loss(out, data.y.long().view(-1))

            if use_unsup_loss:
                aug_data1 = augment(data, aug1)
                aug_data2 = augment(data, aug2)
                out1 = self.model.forward_graphcl(aug_data1)
                out2 = self.model.forward_graphcl(aug_data2)
                unsup_loss = self.model.loss_graphcl(out1, out2)
                loss = sup_loss + lamda * unsup_loss
            else:
                loss = sup_loss

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
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                val_out = self.model(data)
                val_loss = F.nll_loss(val_out, data.y.long().view(-1))
                val_losses.append(val_loss.item())
                y_true += data.y.tolist()
                y_pred += val_out.max(1).indices.tolist()

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
                data = data.to(self.device)
                test_out = self.model(data)
                y_true += data.y.tolist()
                y_pred += test_out.max(1).indices.tolist()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        self.logger.info('Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}'.format(acc, auc, f1))
        return {'acc': acc, 'auc': auc, 'f1': f1}

    def train_process(self):
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

            val_metrics = self.validate_epoch(epoch)
            early_stopping(val_metrics[selection_metric], self.model, epoch, self.best_path)
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
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info('Loaded best checkpoint for testing: {}'.format(best_model_path))

        return self.test()
