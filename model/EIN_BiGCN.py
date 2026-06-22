import sys,os
sys.path.append(os.getcwd())
from utils.dataloader import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
import copy




class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, args, device):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, args, device):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class BiGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_classes, args, device):
        super(BiGCN, self).__init__()
        self.args = args
        self.device = device

        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, args, device)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, args, device)

        self.fc=torch.nn.Linear((out_feats+hid_feats)*2, num_classes)

        self.W_u0 = torch.nn.Linear(1, hid_feats)
        self.W_s0 = torch.nn.Linear(1, hid_feats)
        self.W_d0 = torch.nn.Linear(1, hid_feats)

        self.W_u = torch.nn.Linear(hid_feats, hid_feats)
        self.W_s = torch.nn.Linear(hid_feats, hid_feats)
        self.W_d = torch.nn.Linear(hid_feats, hid_feats)

        self.W_x = torch.nn.Linear(hid_feats*3, (out_feats+hid_feats))

        self.l_u = torch.nn.Linear(hid_feats, 1)
        self.l_s = torch.nn.Linear(hid_feats, 1)
        self.l_d = torch.nn.Linear(hid_feats, 1)

        if self.args.init_alpha  == 'random' and self.args.init_beta == 'random':
            self.raw_alpha = torch.nn.Parameter(torch.rand(1))
            self.raw_beta = torch.nn.Parameter(torch.rand(1))
        else:
            self.raw_alpha = torch.nn.Parameter(torch.tensor(self.args.init_alpha))
            self.raw_beta = torch.nn.Parameter(torch.tensor(self.args.init_beta))

    def init_optimizer(self, args):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        return optimizer
    
    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)  
    
    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)  


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

    def forward(self, data):
        user_state, n_hop = data.user_state, data.num_hop

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

        TD_x = self.TDrumorGCN(data) + xg
        BU_x = self.BUrumorGCN(data) + xg
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x, U, S, D