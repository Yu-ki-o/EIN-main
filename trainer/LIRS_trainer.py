import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch_geometric.loader import DataLoader

from utils.earlystopping import EarlyStopping
from utils.logger import get_logger, get_log_dir


class LIRSTrainer(object):
    def __init__(self, datasets, model, optimizer, args, device):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = device

        train_dataset, val_dataset, test_dataset = datasets
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        self.train_per_epoch = len(self.train_loader)

        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for data in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            out_labels, graph_rep = self.model(data)
            loss = F.nll_loss(out_labels, data.y.view(-1).long())
            loss = loss + self.model.regularization_loss(graph_rep, data)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite LIRS loss detected at epoch {}.'.format(epoch))
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_epoch_loss = train_loss / self.train_per_epoch
        self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))
        return train_epoch_loss

    def validate_epoch(self, epoch):
        val_losses = []
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                val_out, _ = self.model(data)
                val_loss = F.nll_loss(val_out, data.y.view(-1).long())
                val_losses.append(val_loss.item())

        val_loss = np.mean(val_losses)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        return val_loss

    def test(self):
        y_true = []
        y_pred = []
        y_score = []
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                test_out, _ = self.model(data)
                prob = test_out.exp()
                y_true += data.y.view(-1).tolist()
                y_pred += test_out.max(1).indices.tolist()
                y_score += prob[:, 1].tolist()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        true_labels, true_counts = np.unique(y_true, return_counts=True)
        pred_labels, pred_counts = np.unique(y_pred, return_counts=True)
        self.logger.info("Test true label counts: {}".format(
            dict(zip(true_labels.tolist(), true_counts.tolist()))
        ))
        self.logger.info("Test pred label counts: {}".format(
            dict(zip(pred_labels.tolist(), pred_counts.tolist()))
        ))
        self.logger.info("Test positive-score min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
            y_score.min(), y_score.mean(), y_score.max()
        ))
        self.logger.info("Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}".format(acc, auc, f1))
        return {'acc': acc, 'auc': auc, 'f1': f1}

    def train_process(self):
        start_time = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.n_epochs):
            train_epoch_loss = self.train_epoch(epoch)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_loss = self.validate_epoch(epoch)
            early_stopping(val_loss, self.model, epoch, self.best_path)

            if early_stopping.early_stop:
                self.logger.info("Validation performance didn't improve for {} epochs. Training stops.".format(
                    self.args.patience
                ))
                break

        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60),
                        -early_stopping.best_score,
                        early_stopping.best_epoch))

        best_model_path = self.best_path + '.m'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info("Loaded best checkpoint for testing: {}".format(best_model_path))

        return self.test()
