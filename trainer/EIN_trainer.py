import torch.nn.functional as F
import numpy as np
import torch
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

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        self.train_per_epoch = len(self.train_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
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
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data.to(self.device)
            out_labels, U, S, D = self.model(data)
            
            p_loss = self.model.physics_loss(U, S, D, data.user_state)

            loss = F.nll_loss(out_labels, data.y) + p_loss
            if hasattr(self.model, 'auxiliary_loss'):
                aux_loss = self.model.auxiliary_loss()
                if aux_loss is not None:
                    loss = loss + aux_loss
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_epoch_loss = train_loss/self.train_per_epoch

        self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss

    def validate_epoch(self, epoch):

        val_losses = []
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data.to(self.device)
                val_out, _, _, _ = self.model(data)
                val_loss  = F.nll_loss(val_out, data.y)
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
        # test
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data.to(self.device)
                test_out, _, _, _ = self.model(data)

                y_true += data.y.tolist()
                y_pred += test_out.max(1).indices.tolist()

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            self.logger.info("Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}".format(acc, auc, f1))
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

            # validation
            val_metrics = self.validate_epoch(epoch)
            
            early_stopping(val_metrics[selection_metric], self.model, epoch, self.best_path)
            
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
