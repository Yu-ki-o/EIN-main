import sys,os
sys.path.append(os.getcwd())
from utils.dataloader import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm1d, Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from functools import partial
import copy


    
class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_norm (bool, optional): whether or not to normalize adj matrix.
            (default: :obj:`True`)
        gfn (bool, optional): If `True`, only linear transform (1x1 conv) is
            applied to every nodes. (default: :obj:`False`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        edge_index = edge_index[0]
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, dataset=None, num_classes=2, hidden=128, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True, args=None, device=None):
        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.num_classes = num_classes
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        self.args = args
        self.device = device

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
       
        self.use_xg = True

        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = dataset.num_features
        
        self.bn_feat = BatchNorm1d(hidden_in)
        feat_gfn = True  # set true so GCNConv is feat transform
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=feat_gfn)
        if "gating" in global_pool:
            self.gating = torch.nn.Sequential(
                Linear(hidden, hidden),
                torch.nn.ReLU(),
                Linear(hidden, 1),
                torch.nn.Sigmoid())
        else:
            self.gating = None
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
            
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, self.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

        self.W_u0 = torch.nn.Linear(1, hidden)
        self.W_s0 = torch.nn.Linear(1, hidden)
        self.W_d0 = torch.nn.Linear(1, hidden)

        self.W_u = torch.nn.Linear(hidden, hidden)
        self.W_s = torch.nn.Linear(hidden, hidden)
        self.W_d = torch.nn.Linear(hidden, hidden)

        self.W_x = torch.nn.Linear(hidden*3, hidden)

        self.l_u = torch.nn.Linear(hidden, 1)
        self.l_s = torch.nn.Linear(hidden, 1)
        self.l_d = torch.nn.Linear(hidden, 1)

        if self.args.init_alpha  == 'random' and self.args.init_beta == 'random':
            self.raw_alpha = torch.nn.Parameter(torch.rand(1))
            self.raw_beta = torch.nn.Parameter(torch.rand(1))
        else:
            self.raw_alpha = torch.nn.Parameter(torch.tensor(self.args.init_alpha))
            self.raw_beta = torch.nn.Parameter(torch.tensor(self.args.init_beta))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha) 
    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)  

    def forward(self, data):
        x, edge_index, batch, user_state, n_hop = data.x, data.edge_index, data.batch, data.user_state, data.num_hop


        if self.use_xg:
  
            u = torch.sum(user_state, dim=(1, 2)) # just count the number of nodes in a tree excluding root
            s = torch.zeros(user_state.shape[0]).to(self.device) # based on current batch size
            d = torch.zeros(user_state.shape[0]).to(self.device)

            u_0 = u.unsqueeze(1)
            s_0 = s.unsqueeze(1)
            d_0 = d.unsqueeze(1)

            U_ = self.W_u0(u_0) # (64, 1)
            S_ = self.W_s0(s_0)
            D_ = self.W_d0(d_0)

            Ul=[]
            Sl=[]
            Dl=[]

            for i in range(self.args.max_hop):
                U_ = U_ - self.alpha*U_ - self.beta*U_
                U_ = self.W_u(U_)

                S_ = S_ + self.alpha*U_
                S_ = self.W_s(S_)

                D_ = D_ + self.beta*U_
                D_ = self.W_d(D_)
            
                Ul.append(U_)
                Sl.append(S_)
                Dl.append(D_)

            U = torch.stack(Ul, dim=1) # (n, l, h)
            S = torch.stack(Sl, dim=1)
            D = torch.stack(Dl, dim=1)

            hop_ind = (n_hop - 1).long().reshape(user_state.shape[0], 1, 1).expand(-1, -1, self.args.hidden_dim) # n_hop as index

            # find real max-hop for each sample in batch
            U_m = torch.gather(U, 1, hop_ind).reshape(user_state.shape[0], self.args.hidden_dim)
            S_m = torch.gather(S, 1, hop_ind).reshape(user_state.shape[0], self.args.hidden_dim)
            D_m = torch.gather(D, 1, hop_ind).reshape(user_state.shape[0], self.args.hidden_dim)

            xg = torch.cat((U_m, S_m, D_m), dim=1) # (n, h*3)

            xg = self.W_x(xg)

            U = self.l_u(U)
            S = self.l_s(S)
            D = self.l_d(D)

        out = self.forward_BNConvReLU(x, edge_index, batch, xg)

        return out, U, S, D
       

    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
 
        x = x if xg is None else x + xg

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def physics_loss(self, U, S, D, true_state):
        pred_states = torch.stack((U, S, D), dim=2)
        logp_state = F.log_softmax(pred_states, dim=2)

        sum_t_state = true_state.sum(dim=-1, keepdim=True) # can be mask

        p_true_state = (true_state / sum_t_state)

        p_true_state[torch.isnan(p_true_state)] = 0 # masking

        mask = torch.where(sum_t_state != 0, torch.tensor(1), torch.tensor(0))

        logp_state = (mask.unsqueeze(-2) * logp_state).squeeze(-1) # masking

        kl_div = F.kl_div(logp_state, p_true_state, reduction='none')

        batch_p_loss = kl_div.sum(dim=(1, 2), keepdim=True).mean()

        return self.args.lamda * batch_p_loss


    def init_optimizer(self, args):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        return optimizer

    def __repr__(self):
        return self.__class__.__name__
