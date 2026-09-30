"""KPG graph generator and EIN adapter.

Ported from https://github.com/kkkkk001/KPG at
7b41f6647fba7f23b8d461a85df33bdf4c699ce8 (generators.py,
KPG_main_final.py, utils.py, train_gcn.py), Yusong Zhang et al., WWW 2025.

The author token CVAE is retained below as TokenCVAE. EIN datasets expose
sentence vectors, not the author's word IDs / 5000-dimensional word counts:
the active CVAE therefore reconstructs continuous features with MSE. The
TD/BU GCN, parent-aware ENS, local/global search, root fallback, modified
rollout, reward-weighted losses and cumulative patience follow the author
pipeline. Compact per-tree tensors replace its padded candidate buffers.
The separate BERT ensemble is NOT part of this feature-interface adapter.
See docs/KPG.md for stage scheduling and deliberate implementation fixes.
"""

from dataclasses import dataclass
import hashlib
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


@torch.no_grad()
def _sample_with_replacement(weights, num_samples, generator=None):
    """Keep strict determinism without CUDA multinomial's multi-draw cumsum.

    In strict mode, compute the inverse CDF on CPU. Uniform draws still use
    the original device/generator, preserving the per-event random stream.
    Single draws retain PyTorch's original sampling path.
    """
    if num_samples <= 1 or not torch.are_deterministic_algorithms_enabled():
        return torch.multinomial(weights, num_samples, replacement=True,
                                 generator=generator)
    probabilities = weights.detach().to(device='cpu', dtype=torch.float64)
    if probabilities.ndim != 1 or not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError('KPG sampling requires finite nonnegative 1-D weights.')
    cumulative = probabilities.cumsum(0)
    if cumulative.numel() == 0 or not bool(cumulative[-1] > 0):
        raise ValueError('KPG sampling requires a positive total weight.')
    cumulative = cumulative / cumulative[-1]
    uniform = torch.rand(num_samples, device=weights.device,
                         dtype=torch.float64, generator=generator).cpu()
    # right=True skips zero-mass bins, including when a uniform draw is zero.
    indices = torch.searchsorted(cumulative, uniform, right=True)
    return indices.to(device=weights.device)


class CVAE(nn.Module):
    """Continuous-feature counterpart of the author's context/root CVAE."""

    def __init__(self, in_feats, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.root_encoder = nn.Linear(in_feats, hidden_dim)
        self.context_encoder = nn.Linear(in_feats, hidden_dim)
        self.response_encoder = nn.Linear(in_feats, hidden_dim)
        self.posterior = nn.Linear(3 * hidden_dim, 2 * latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + latent_dim, hidden_dim),
            nn.ELU(), nn.Linear(hidden_dim, in_feats),
        )
        self.latent_dim = latent_dim

    def condition(self, context, root):
        return torch.cat((self.root_encoder(root), self.context_encoder(context)), -1)

    def forward(self, response, context, root):
        condition = self.condition(context, root)
        mu, logvar = self.posterior(torch.cat(
            (self.response_encoder(response), condition), -1
        )).chunk(2, -1)
        logvar = logvar.clamp(-12, 12)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return z, mu, logvar, self.decoder(torch.cat((z, condition), -1))

    def generate(self, context, root, generator=None):
        z = torch.randn((context.size(0), self.latent_dim), device=context.device,
                        dtype=context.dtype, generator=generator)
        return self.decoder(torch.cat((z, self.condition(context, root)), -1))


class GCN(nn.Module):
    """Author reward/final classifier: separate two-layer TD and BU GCNs."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_class, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.conv3 = GCNConv(input_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(2 * output_dim, num_class)
        self.dropout = dropout

    def forward(self, data):
        edge = getattr(data, 'directed_edge_index', None)
        if edge is None:
            edge = data.edge_index
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)
        td = F.dropout(F.elu(self.conv1(data.x, edge)), self.dropout, self.training)
        td = global_mean_pool(F.elu(self.conv2(td, edge)), batch)
        bu = F.dropout(F.elu(self.conv3(data.x, edge.flip(0))), self.dropout, self.training)
        bu = global_mean_pool(F.elu(self.conv4(bu, edge.flip(0))), batch)
        return F.log_softmax(self.fc(torch.cat((td, bu), -1)), -1)


pretrainedGCN = GCN


class EndNodeSelector(nn.Module):
    """Author ENS layers; probabilities are normalized within one event."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(2 * hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = dropout

    def forward(self, pool_x, parents, selected, edges, candidates):
        # Like the original padded representation, selected nodes form the
        # current graph and candidate copies only have GCN self-loops.
        n = len(selected)
        x = torch.cat((pool_x[selected], pool_x), 0)
        h = self.conv1(x, edges)
        parent_h = torch.zeros_like(h)
        mapping = {node: i for i, node in enumerate(selected)}
        for child, parent in enumerate(parents.tolist()):
            if parent in mapping:
                parent_h[n + child] = h[mapping[parent]]
        h = F.dropout(F.elu(torch.cat((h, parent_h), -1)),
                      self.dropout, self.training)
        scores = self.fc(F.elu(self.conv2(h, edges))).squeeze(-1)
        return F.log_softmax(scores[n + candidates], dim=0)


@dataclass
class _TreeState:
    x: torch.Tensor
    parents: torch.Tensor
    selected: list
    edges: torch.Tensor
    original_size: int


class KPG(nn.Module):
    """EIN model API: forward(Data/Batch) -> (log_probs, zero, zero).

    Call prepare through KPGTrainer before evaluating a fresh model. Complete
    state_dict checkpoints contain all three networks and the readiness flags.
    Eval construction never reads labels and uses a per-input random stream.
    """

    def __init__(self, in_feats, hidden_dim, num_classes, args, device=None):
        super().__init__()
        self.args = args
        self.in_feats = int(in_feats)
        self.num_classes = int(num_classes)
        hidden = int(getattr(args, 'kpg_hidden_dim', hidden_dim))
        dropout = float(getattr(args, 'kpg_dropout', 0.5))
        self.response_generator = CVAE(in_feats, hidden, int(getattr(args, 'kpg_latent_dim', 32)))
        self.selector = EndNodeSelector(in_feats, hidden, hidden, dropout)
        self.reward_model = GCN(in_feats, hidden, hidden, num_classes, dropout)
        self.classifier = GCN(in_feats, hidden, hidden, num_classes, dropout)
        self.max_steps = int(getattr(args, 'kpg_max_steps', 200))
        self.candidate_threshold = int(getattr(args, 'kpg_candidate_threshold', 5))
        self.rollout_steps = int(getattr(args, 'kpg_rollout_steps', 10))
        self.max_patience = int(getattr(args, 'kpg_max_patience', 4))
        self.epsilon = float(getattr(args, 'kpg_epsilon', 0.8))
        self.eval_seed = int(getattr(args, 'kpg_eval_seed', 2022))
        budget = int(getattr(args, 'kpg_max_nodes', 0))
        if min(self.max_steps, self.candidate_threshold, self.max_patience) < 1:
            raise ValueError('KPG steps, candidate threshold and patience must be positive.')
        if budget < 0 or self.rollout_steps < 0 or not 0 <= self.epsilon <= 1:
            raise ValueError('Invalid KPG budget, rollout steps or epsilon.')
        self.register_buffer('node_budget', torch.tensor(budget, dtype=torch.long))
        self.register_buffer('reward_ready', torch.tensor(False))
        self.register_buffer('generator_ready', torch.tensor(False))
        self.register_buffer('construction_ready', torch.tensor(False))

    def init_optimizer(self, args):
        # The final classifier has its own optimizer; the trainer creates the
        # reward, ENS and CRG optimizers for their separate stages.
        return torch.optim.Adam(self.classifier.parameters(), lr=args.lr,
                                weight_decay=getattr(args, 'weight_decay', 0.0))

    def train(self, mode=True):
        super().train(mode)
        if bool(self.reward_ready):
            self.reward_model.eval()
        if bool(self.construction_ready):
            self.selector.eval()
            self.response_generator.eval()
        return self

    @staticmethod
    def _graphs(data):
        return data.to_data_list() if isinstance(data, Batch) else [data]

    def configure_budget(self, train_dataset):
        if int(self.node_budget) == 0:
            sizes = [int(graph.num_nodes) for graph in train_dataset]
            if not sizes:
                raise ValueError('KPG needs a nonempty training split.')
            tau = float(getattr(self.args, 'kpg_tau', 4.0))
            if tau <= 0:
                raise ValueError('kpg_tau must be positive.')
            # Training split only: validation/test sizes cannot set the budget.
            self.node_budget.fill_(max(2, int(tau * np.median(sizes))))

    def _initial_state(self, graph):
        x = graph.x.detach()
        if x.ndim != 2 or x.size(1) != self.in_feats or x.size(0) == 0:
            raise ValueError('KPG expects nonempty [nodes, in_feats] sentence features.')
        edge = getattr(graph, 'directed_edge_index', None)
        edge = graph.edge_index if edge is None else edge
        edge = edge.to(device=x.device, dtype=torch.long)
        parents = torch.full((x.size(0),), -1, device=x.device, dtype=torch.long)
        for u, v in edge.t().tolist():
            if u == v:
                continue
            if not (0 <= u < x.size(0) and 0 < v < x.size(0)) or parents[v] != -1:
                raise ValueError('KPG requires a directed reply tree rooted at node 0.')
            parents[v] = u
        for v in range(1, x.size(0)):
            seen = set()
            u = v
            while u != 0:
                if u < 0 or u in seen:
                    raise ValueError('KPG reply tree is disconnected or cyclic.')
                seen.add(u)
                u = int(parents[u])
        return _TreeState(x.clone(), parents, [0], edge.new_empty((2, 0)), x.size(0))

    def _eval_generator(self, state):
        digest = hashlib.sha256()
        digest.update(state.x.cpu().contiguous().numpy().tobytes())
        digest.update(state.parents.cpu().numpy().tobytes())
        seed = (int.from_bytes(digest.digest()[:8], 'little') + self.eval_seed) % (2**63 - 1)
        return torch.Generator(device=state.x.device).manual_seed(seed)

    @staticmethod
    def _graph(state):
        return Data(x=state.x[state.selected], edge_index=state.edges)

    @staticmethod
    def _extend(state, node):
        selected = state.selected + [node]
        parent = int(state.parents[node])
        # Original KPG reconnects global candidates to root when their parent
        # is absent. This is intentionally retained (not path-preserving).
        parent_pos = state.selected.index(parent) if parent in state.selected else 0
        edge = state.edges.new_tensor([[parent_pos], [len(state.selected)]])
        return _TreeState(state.x, state.parents, selected,
                          torch.cat((state.edges, edge), 1), state.original_size)

    def _candidates(self, state, generator, local_only=False):
        available = torch.ones(state.x.size(0), dtype=torch.bool, device=state.x.device)
        available[state.selected] = False
        global_ids = available.nonzero().flatten()
        local = available & torch.isin(state.parents, state.parents.new_tensor(state.selected))
        local_ids = local.nonzero().flatten()
        choose_local = local_only or float(torch.rand((), device=state.x.device, generator=generator)) < self.epsilon
        return local_ids if choose_local and local_ids.numel() else global_ids

    @torch.no_grad()
    def _augment(self, state, generator):
        if state.x.size(0) - len(state.selected) > self.candidate_threshold:
            return state
        if not bool(self.generator_ready):
            raise RuntimeError('Train CRG before enabling candidate augmentation.')
        # Author utils.py weights contexts by branching and node order.
        count = torch.bincount(state.parents[1:], minlength=state.x.size(0)).float()
        active = (count > 0).nonzero().flatten()
        weights = count.new_zeros(count.shape)
        if active.numel():
            order = active.float() + 1
            order = ((1 - order / order.sum()) / (len(active) - 1)
                     if len(active) > 1 else torch.ones_like(order))
            mass = count[active] / count.sum() * order / 2
            weights[active] += mass
            for i, parent in enumerate(active):
                weights[state.parents == parent] += mass[i] / count[parent]
        else:
            weights[0] = 1
        n = self.candidate_threshold + 1 - (state.x.size(0) - len(state.selected))
        contexts = _sample_with_replacement(weights, n, generator=generator)
        generated = self.response_generator.generate(
            state.x[contexts], state.x[0].expand(n, -1), generator)
        return _TreeState(torch.cat((state.x, generated), 0),
                          torch.cat((state.parents, contexts), 0),
                          state.selected, state.edges, state.original_size)

    @torch.no_grad()
    def calculate_reward(self, state, generator):
        reward = self.reward_model(self._graph(state)).exp()[0]
        future = []
        for _ in range(self.rollout_steps):
            if len(state.selected) >= int(self.node_budget):
                break
            ids = self._candidates(state, generator, local_only=True)
            if ids.numel() == 0:
                break
            logp = self.selector(state.x, state.parents, state.selected, state.edges, ids)
            idx = torch.multinomial(logp.exp(), 1, generator=generator).item()
            state = self._extend(state, int(ids[idx]))
            future.append(self.reward_model(self._graph(state)).exp()[0])
        return reward + torch.stack(future).mean(0) if future else reward

    def construct_graph(self, graph, policy_training=False):
        if not bool(self.reward_ready) or int(self.node_budget) < 1:
            raise RuntimeError('KPG reward pretraining and node-budget setup are required.')
        state = self._initial_state(graph)
        generator = None if policy_training else self._eval_generator(state)
        with torch.no_grad():
            if policy_training:
                reference = int(graph.y.view(-1)[0])
            else:
                # Labels are deliberately never accessed in inference.
                reference = int(self.reward_model(Data(
                    x=state.x, edge_index=torch.stack((state.parents[1:],
                        torch.arange(1, state.x.size(0), device=state.x.device)))
                )).argmax(-1)[0])
            previous = self.calculate_reward(state, generator)
        patience = torch.zeros(self.num_classes, dtype=torch.long, device=state.x.device)
        losses = []
        for _ in range(self.max_steps):
            if len(state.selected) >= int(self.node_budget):
                break
            state = self._augment(state, generator)
            ids = self._candidates(state, generator)
            if not ids.numel():
                break
            with torch.set_grad_enabled(policy_training):
                logp = self.selector(state.x, state.parents, state.selected, state.edges, ids)
            idx = torch.multinomial(logp.detach().exp(), 1, generator=generator).item()
            trial = self._extend(state, int(ids[idx]))
            with torch.no_grad():
                rewards = self.calculate_reward(trial, generator)
                delta = rewards - previous
                patience += (delta < 0).long()
                score = self.reward_model(self._graph(trial)).exp()[0, reference]
            if policy_training:
                # Original main script: CE(action) * exp(-reward_diff) * (1.5-p_y).
                losses.append(-logp[idx] * (-delta[reference]).exp() * (1.5 - score))
            if not policy_training or delta[reference] >= 0:
                state = trial
            # Original keeps the reference reward on a rejected decrease.
            rewards[reference] = torch.maximum(rewards[reference], previous[reference])
            previous = rewards
            others = torch.arange(self.num_classes, device=state.x.device) != reference
            if patience[reference] >= self.max_patience and bool((patience[others] > 0).all()):
                break
        result = self._graph(state)
        result.original_node_id = torch.tensor(
            [v if v < state.original_size else -1 for v in state.selected],
            device=state.x.device, dtype=torch.long)
        result.is_generated = result.original_node_id < 0
        result.kpg_prebuilt = torch.tensor([True], device=state.x.device)
        loss = torch.stack(losses).mean() if losses else self.selector.fc.weight.sum() * 0
        return result, loss

    def forward(self, data):
        if not bool(self.construction_ready):
            raise RuntimeError('Use KPGTrainer.train_process() or load a complete KPG checkpoint first.')
        prebuilt = getattr(data, 'kpg_prebuilt', None)
        if prebuilt is not None and bool(prebuilt.all()):
            key_batch = data
        else:
            with torch.no_grad():
                graphs = [self.construct_graph(g)[0] for g in self._graphs(data)]
                key_batch = Batch.from_data_list(graphs)
        out = self.classifier(key_batch)
        zero = out.sum() * 0
        return out, zero, zero

    def generator_loss(self, graph, reward_weighted=True):
        edge = graph.edge_index
        if edge.numel() == 0:
            # Author also reconstructs the root; handle root-only events.
            parent = child = torch.zeros(1, dtype=torch.long, device=graph.x.device)
        else:
            parent, child = edge
        _, mu, logvar, decoded = self.response_generator(
            graph.x[child], graph.x[parent], graph.x[0].expand(child.numel(), -1))
        kl = -0.5 * (1 + logvar - logvar.exp() - mu.square()).sum(-1).mean()
        loss = 0.5 * F.mse_loss(decoded, graph.x[child]) + float(
            getattr(self.args, 'kpg_kl_weight', 1.0)) * kl
        if reward_weighted:
            with torch.no_grad():
                y = int(graph.y.view(-1)[0])
                before = self.reward_model(graph).exp()[0, y]
                x = graph.x.clone()
                x[child] = decoded.detach()
                after = self.reward_model(Data(x=x, edge_index=edge)).exp()[0, y]
                weight = (before - after).exp()
            loss = loss * weight
        return loss


# Trainer is kept here intentionally: this file contains the KPG-specific
# stages rather than distributing author main-script logic across the project.
from trainer.EBGCN_trainer import EBGCNTrainer


class KPGTrainer(EBGCNTrainer):
    """Reward pretrain -> CRG warmup -> alternating ENS/CRG -> final GCN."""

    def __init__(self, datasets, model, optimizer, args, device):
        super().__init__(datasets, model, optimizer, args, device)
        self.datasets = datasets
        os.makedirs(args.log_dir, exist_ok=True)
        self.model.configure_budget(datasets[0])

    def _step(self, loss, optimizer):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite KPG stage loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(parameters, float(getattr(self.args, 'kpg_grad_clip', 5.0)))
        optimizer.step()

    def prepare(self):
        model = self.model
        lr = float(getattr(self.args, 'kpg_lr', self.args.lr))
        weight_decay = float(getattr(self.args, 'weight_decay', 0.0))
        reward_optimizer = torch.optim.Adam(model.reward_model.parameters(), lr=lr, weight_decay=weight_decay)
        ens_optimizer = torch.optim.Adam(model.selector.parameters(), lr=lr, weight_decay=weight_decay)
        crg_optimizer = torch.optim.Adam(model.response_generator.parameters(), lr=lr, weight_decay=weight_decay)
        pretrain_epochs = int(getattr(self.args, 'kpg_reward_epochs', 30))
        warmup_epochs = int(getattr(self.args, 'kpg_crg_warmup_epochs', 5))
        alternating_epochs = int(getattr(self.args, 'kpg_alternating_epochs', 5))
        if min(pretrain_epochs, warmup_epochs, alternating_epochs) < 1:
            raise ValueError('KPG training stages must each have at least one epoch.')
        for epoch in range(pretrain_epochs):
            model.reward_model.train()
            total = 0.0
            for data in self.train_loader:
                data = self._move_to_device(data)
                loss = F.nll_loss(model.reward_model(data), data.y.view(-1).long())
                self._step(loss, reward_optimizer)
                total += float(loss.detach())
            self.logger.info('KPG reward pretrain %d/%d: %.6f', epoch + 1, pretrain_epochs, total / len(self.train_loader))
        model.reward_ready.fill_(True)
        model.reward_model.requires_grad_(False).eval()
        train_graphs = sorted(self.datasets[0], key=lambda g: g.num_nodes, reverse=True)
        for epoch in range(warmup_epochs):
            model.response_generator.train()
            total = 0.0
            for graph in train_graphs:
                graph = graph.clone().to(self.device)
                state = model._initial_state(graph)
                clean = Data(x=state.x, edge_index=torch.stack((state.parents[1:],
                    torch.arange(1, state.x.size(0), device=self.device))), y=graph.y)
                loss = model.generator_loss(clean, reward_weighted=False)
                self._step(loss, crg_optimizer)
                total += float(loss.detach())
            self.logger.info('KPG CRG warmup %d/%d: %.6f', epoch + 1, warmup_epochs, total / len(train_graphs))
        model.generator_ready.fill_(True)
        for epoch in range(alternating_epochs):
            ens_total = crg_total = 0.0
            for graph in train_graphs:
                graph = graph.clone().to(self.device)
                model.selector.train()
                model.response_generator.eval()
                key, policy_loss = model.construct_graph(graph, policy_training=True)
                self._step(policy_loss, ens_optimizer)
                model.selector.eval()
                model.response_generator.train()
                # Learn from ENS-refined observed pairs, not invented targets.
                real_edges = key.edge_index[:, ~key.is_generated[key.edge_index[0]] & ~key.is_generated[key.edge_index[1]]]
                real_nodes = (~key.is_generated).nonzero().flatten()
                remap = key.original_node_id.new_full((key.num_nodes,), -1)
                remap[real_nodes] = torch.arange(real_nodes.numel(), device=self.device)
                refined = Data(x=key.x[real_nodes].detach(),
                               edge_index=remap[real_edges], y=graph.y)
                crg_loss = model.generator_loss(refined)
                self._step(crg_loss, crg_optimizer)
                ens_total += float(policy_loss.detach())
                crg_total += float(crg_loss.detach())
            self.logger.info('KPG alternating %d/%d: ENS %.6f | CRG %.6f', epoch + 1,
                             alternating_epochs, ens_total / len(train_graphs), crg_total / len(train_graphs))
        model.selector.requires_grad_(False).eval()
        model.response_generator.requires_grad_(False).eval()
        model.construction_ready.fill_(True)

    def _cache_graphs(self):
        from torch_geometric.loader import DataLoader
        loaders = []
        self.model.eval()
        for split, dataset in zip(('train', 'val', 'test'), self.datasets):
            graphs = []
            with torch.no_grad():
                for graph in dataset:
                    key, _ = self.model.construct_graph(graph.clone().to(self.device))
                    key.y = graph.y.to(self.device).clone()
                    graphs.append(key.cpu())
            loaders.append(DataLoader(graphs, batch_size=self.args.batch_size,
                                      shuffle=split == 'train', **self._loader_kwargs()))
            self.logger.info('KPG cached %s: %d graphs', split, len(graphs))
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.train_per_epoch = len(self.train_loader)

    def train_process(self):
        checkpoint = getattr(self.args, 'kpg_checkpoint', None)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
            if not bool(self.model.construction_ready):
                raise ValueError('kpg_checkpoint must be a complete KPG state_dict.')
        elif bool(getattr(self.args, 'kpg_test_only', False)):
            raise ValueError('kpg_test_only requires kpg_checkpoint.')
        else:
            self.prepare()
        if bool(getattr(self.args, 'kpg_test_only', False)):
            return self.test()
        self._cache_graphs()
        return super().train_process()

    def _evaluate(self, loader):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        self.model.eval()
        labels, probabilities, losses = [], [], []
        with torch.no_grad():
            for data in loader:
                data = self._move_to_device(data)
                out, _, _ = self.model(data)
                labels.extend(data.y.view(-1).tolist())
                probabilities.extend(out.exp().cpu().tolist())
                losses.append((F.nll_loss(out, data.y.view(-1).long(), reduction='sum').item(), out.size(0)))
        if not labels:
            raise ValueError('KPG evaluation split is empty.')
        prob = np.asarray(probabilities)
        pred = prob.argmax(1)
        try:
            auc = roc_auc_score(labels, prob[:, 1]) if prob.shape[1] == 2 else roc_auc_score(
                labels, prob, multi_class='ovr', labels=list(range(prob.shape[1])))
        except ValueError:
            auc = float('nan')
        return dict(val_loss=sum(x for x, _ in losses) / len(labels),
                    val_acc=accuracy_score(labels, pred), val_auc=auc,
                    val_f1=f1_score(labels, pred, average='binary' if prob.shape[1] == 2 else 'macro', zero_division=0))

    def validate_epoch(self, epoch):
        metrics = self._evaluate(self.val_loader)
        self.logger.info('KPG validation epoch %d: %s', epoch, metrics)
        return metrics

    def test(self):
        metrics = self._evaluate(self.test_loader)
        result = {name: metrics['val_' + name] for name in ('acc', 'auc', 'f1')}
        self.logger.info('KPG test: %s', result)
        return result

# Original token-based CRG, retained for readers or datasets with author word IDs.
# Only device ownership, class names and None comparisons are adapted.
SOS_token = 5000


class _TokenDeviceModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

class TokenEncoder(_TokenDeviceModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(TokenEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*3, output_size*2)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps*std

    def encode_context(self, context, root):
        if context is not None:
            context = self.embed(context).unsqueeze(1)
            context, _ = self.gru(context)
            context = context[-1].squeeze(1)
            if self.bidirectional:
                context = context[:,self.hidden_size:]+context[:,:self.hidden_size]
            context = torch.cat((root, context), dim=1)
        else:
            context = torch.cat((root, torch.zeros_like(root).to(self.device)), dim=1)
        return context
    
    def forward(self, response, context, root):
        response = self.embed(response).unsqueeze(1)
        response, _ = self.gru(response)
        response = response[-1].squeeze(1)
        if self.bidirectional:
            response = response[:,self.hidden_size:]+response[:,:self.hidden_size]
        context = self.encode_context(context, root)
        output = torch.cat((response, context), dim=1)
        output = self.fc(output)
        mu, logvar = torch.chunk(output, 2, dim=1)
        z = self.sample(mu, logvar)
        return z, context, mu, logvar

class TokenDecoder(_TokenDeviceModule):
    def __init__(self, z_size, context_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(TokenDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + z_size + context_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(z_size + context_size, hidden_size)
        self.fc2 = nn.Linear(z_size + context_size + hidden_size, output_size)
    
    def forward(self, z, n_words, response, context, truthrate=1):
        decoded_r = torch.zeros(n_words, 1, self.output_size).to(self.device)
        word_i = torch.LongTensor([SOS_token]).to(self.device)
        encoded_r = torch.cat((z, context), dim=1)
        h0 = self.fc1(encoded_r).unsqueeze(0)
        if self.bidirectional:
            h0 = torch.cat((h0,h0),dim=0)
        for i in range(1,n_words):
            decoded_r_i, h0 = self.cal_word_i(encoded_r, word_i, h0)
            decoded_r[i] = decoded_r_i
            # set_trace()
            word_i = decoded_r_i.topk(2)[1]
            if response is not None:
                if np.random.rand() < truthrate:
                    word_i = response[i]
                else:
                    word_i = word_i[:,0] if word_i[:,0] != SOS_token else word_i[:,1]
            else:
                word_i = word_i[:,0] if word_i[:,0] != SOS_token else word_i[:,1]
            word_i = torch.LongTensor([word_i]).to(self.device)
        return decoded_r.squeeze(1)[1:, :self.output_size-1]

    def cal_word_i(self, encoded_r, word_i, h0):
        word_i = self.embed(word_i)
        word_i = torch.cat([word_i, encoded_r], 1).unsqueeze(0)
        output, h0 = self.gru(word_i, h0)
        output = output.squeeze(0)
        if self.bidirectional:
            output = output[:,self.hidden_size:]+output[:,:self.hidden_size]
        output = torch.cat((output, encoded_r), 1)
        output = self.fc2(output)
        return output, h0 

class TokenCVAE(_TokenDeviceModule):
    def __init__(self, vocab_size, encoder_hidden_size, decoder_hidden_size, z_size, num_elayers=1, num_dlayers=1, bidirectional=True):
        super(TokenCVAE, self).__init__()
        self.z_size = z_size
        self.encoder = TokenEncoder(vocab_size, encoder_hidden_size, z_size, num_elayers, bidirectional)
        self.decoder = TokenDecoder(z_size, encoder_hidden_size*2, decoder_hidden_size, vocab_size, num_dlayers, bidirectional)
        self.RootEmbd = nn.Embedding(vocab_size, encoder_hidden_size)
        self.RootGRU = nn.GRU(encoder_hidden_size, encoder_hidden_size, bidirectional=bidirectional)
        self.encoder_hidden_size = encoder_hidden_size
        self.bidirectional = bidirectional

    def forward(self, response, context, root, truthrate=1):
        root = self.RootEmbd(root).unsqueeze(1)
        root, _ = self.RootGRU(root)
        root = root[-1].squeeze(1)
        if self.bidirectional:
            root = root[:,self.encoder_hidden_size:]+root[:,:self.encoder_hidden_size]
        z, encoded_c, mu, logvar = self.encoder(response, context, root)
        decoded = self.decoder(z, response.size(0), response, encoded_c, truthrate)
        return z, mu, logvar, decoded

    def generate(self, n_words, context, root):
        root = self.RootEmbd(root).unsqueeze(1)
        root, _ = self.RootGRU(root)
        root = root[-1].squeeze(1)
        if self.bidirectional:
            root = root[:,self.encoder_hidden_size:]+root[:,:self.encoder_hidden_size]
        context = self.encoder.encode_context(context, root)
        z = torch.randn((1, self.z_size)).to(self.device)
        decoded = self.decoder(z, n_words, None, context)
        return decoded
    


