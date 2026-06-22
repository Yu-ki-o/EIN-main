import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, mode='min', metric_name='val_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            mode (str): 'min' for metrics where lower is better, 'max' otherwise.
            metric_name (str): Name of the monitored metric for logging/errors.
        """
        if mode not in {'min', 'max'}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max', got {}".format(mode))
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.metric_name = metric_name
        self.counter = 0
        self.best_epoch = 0
        self.best_score = None
        self.best_value = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, metric_value, model, epoch, best_path):

        if not np.isfinite(metric_value):
            raise ValueError(
                "Selection metric {} is not finite: {}".format(
                    self.metric_name, metric_value
                )
            )

        score = -metric_value if self.mode == 'min' else metric_value

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_value = metric_value
            self.save_checkpoint(metric_value, model, best_path)
        elif score <= self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_value = metric_value
            self.save_checkpoint(metric_value, model, best_path)
            self.counter = 0

    def save_checkpoint(self, metric_value, model, best_path):
        '''Saves model when the monitored validation metric improves.'''
        torch.save(model.state_dict(), best_path +'.m')
        print(
            'Best {} updated to {:.6f}, saving current best model...'.format(
                self.metric_name, metric_value
            )
        )
        if self.metric_name == 'val_loss':
            self.val_loss_min = metric_value
