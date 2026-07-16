import torch
import torch.nn.functional as F

from trainer.EBGCN_trainer import EBGCNTrainer


class EBGCNStateAuxSameDiffTrainer(EBGCNTrainer):
    """Separate trainer that adds dual-subgraph auxiliary supervision."""

    def __init__(self, datasets, model, optimizer, args, device):
        super().__init__(datasets, model, optimizer, args, device)
        self.state_aux_loss_weight = float(
            getattr(args, 'ebgcn_state_aux_loss_weight', 1.0)
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_edge_loss = 0.0
        total_state_aux_loss = 0.0

        for data in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            data = self._move_to_device(data)
            out, td_edge_loss, bu_edge_loss = self.model(data)
            cls_loss = F.nll_loss(out, data.y.view(-1).long())
            edge_loss = self._edge_loss(td_edge_loss, bu_edge_loss, out)
            state_aux_loss = self.model.auxiliary_loss()
            loss = (
                cls_loss
                + self.edge_loss_weight * edge_loss
                + self.state_aux_loss_weight * state_aux_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    'Non-finite EBGCN dual-subgraph loss at epoch {}.'.format(epoch)
                )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_edge_loss += edge_loss.item()
            total_state_aux_loss += state_aux_loss.item()

        average_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '*******Training Epoch {}: Loss {:.6f} | Class {:.6f} | '
            'Edge {:.6f} | StateAux {:.6f}'.format(
                epoch,
                average_loss,
                total_cls_loss / self.train_per_epoch,
                total_edge_loss / self.train_per_epoch,
                total_state_aux_loss / self.train_per_epoch,
            )
        )
        return average_loss
