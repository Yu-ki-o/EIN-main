import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class HSIC(nn.Module):
    def __init__(self, kernel_method='rbf', sigma=0.1, device='cpu'):
        super().__init__()
        self.kernel_method = kernel_method
        self.sigma = sigma
        self.device = device

    def compute_centering_matrix(self, n):
        eye = torch.eye(n, device=self.device)
        ones = torch.ones((n, n), device=self.device)
        return eye - ones / n

    def compute_kernel(self, x, y=None):
        if y is None:
            y = x
        if self.kernel_method == 'linear':
            return torch.matmul(x, y.t())
        if self.kernel_method == 'rbf':
            dist = torch.cdist(x, y, p=2).pow(2).clamp_min(0.0)
            sigma_sq = max(self.sigma ** 2, 1e-12)
            exponent = (-0.5 * dist / sigma_sq).clamp(min=-50.0, max=0.0)
            return torch.exp(exponent)
        raise ValueError('Unsupported kernel method: {}'.format(self.kernel_method))

    def forward(self, x, y):
        n = x.shape[0]
        if n <= 1:
            return x.new_tensor(0.0)
        center = self.compute_centering_matrix(n)
        kx = self.compute_kernel(x)
        ky = self.compute_kernel(y)
        return torch.trace(torch.matmul(torch.matmul(kx, center), torch.matmul(ky, center))) / ((n - 1) ** 2)


class LIRSGIN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.num_classes = num_classes
        self.num_domains = len(getattr(args, 'ood_source_datasets', []) or [])
        self.num_domains = max(self.num_domains, 1)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.pooling = getattr(args, 'global_pool', 'sum')

        n_layers = getattr(args, 'n_layers_conv', 3)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer_idx in range(n_layers):
            input_dim = in_feats if layer_idx == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.hsic = HSIC(
            kernel_method=getattr(args, 'lirs_kernel_method', 'rbf'),
            sigma=getattr(args, 'lirs_sigma', 0.1),
            device=device
        )

    def init_optimizer(self, args):
        return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def encode(self, data):
        x = data.x.float()
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == 'mean':
            return global_mean_pool(x, data.batch)
        return global_add_pool(x, data.batch)

    def domain_proxy(self, data):
        if not hasattr(data, 'domain_id'):
            return None
        domain_id = data.domain_id.view(-1).long().to(self.device)
        domain_id = torch.clamp(domain_id, min=0, max=self.num_domains - 1)
        return F.one_hot(domain_id, num_classes=self.num_domains).float()

    def forward(self, data):
        graph_rep = self.encode(data)
        logits = self.classifier(graph_rep)
        return F.log_softmax(logits, dim=-1), graph_rep

    def regularization_loss(self, graph_rep, data):
        domain_proxy = self.domain_proxy(data)
        if domain_proxy is None or domain_proxy.shape[1] <= 1:
            return graph_rep.new_tensor(0.0)
        if getattr(self.args, 'lirs_normalize_hsic_inputs', True):
            graph_rep = F.normalize(graph_rep, p=2, dim=1)
            domain_proxy = F.normalize(domain_proxy, p=2, dim=1)
        return getattr(self.args, 'lirs_hsic_penalty', 0.1) * self.hsic(graph_rep, domain_proxy)
