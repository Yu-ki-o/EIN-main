"""Plain graph-classification GCN with the existing EIN trainer interface."""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, args):
        super().__init__()
        num_layers = int(getattr(args, 'n_layers_conv', 2))
        if num_layers < 1:
            raise ValueError('n_layers_conv must be at least 1')
        pool = getattr(args, 'global_pool', 'mean')
        if pool not in {'mean', 'sum'}:
            raise ValueError('global_pool must be mean or sum')
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.dropout = float(getattr(args, 'dropout', 0.5))
        self.max_hop = int(getattr(args, 'max_hop', 1))
        self.convs = nn.ModuleList([
            GCNConv(in_feats if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        hidden = data.x.float()
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        for conv in self.convs:
            hidden = F.relu(conv(hidden, data.edge_index))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        output = F.log_softmax(self.classifier(self.pool(hidden, batch)), dim=-1)
        # Placeholders only: this baseline has no state-prediction branch.
        dummy = output.new_zeros(output.size(0), self.max_hop, 1)
        return output, dummy, dummy, dummy

    def physics_loss(self, U, S, D, true_state):
        return self.classifier.weight.new_zeros(())

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
