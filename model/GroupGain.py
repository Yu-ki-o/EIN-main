"""Context-constrained group graphs and conditional discriminative gain.

``GroupGain.forward`` returns logits. ``EINGroupGain`` exposes the existing
EIN trainer's four-output interface, with CrossEntropyLoss on those logits.
Teachers are external, frozen training utilities; checkpoints and inference
never depend on a teacher, labels, or stored contribution targets.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from .group_gain_grouping import (
    DEFAULT_RELATION_MAPPING, GroupGraph, build_group_graph,
)


@dataclass
class GainTargets:
    """Signed CE differences; samples are (event index, group, context IDs)."""

    samples: list
    values: torch.Tensor
    diagnostics: dict


class RelationResidualLayer(nn.Module):
    """Separate self transform and forward/inverse relation transforms."""

    def __init__(self, hidden_dim, num_relations, dropout):
        super().__init__()
        self.self_transform = nn.Linear(hidden_dim, hidden_dim)
        self.relations = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(2 * num_relations)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, edge_index, edge_type, sender_gate):
        update = self.self_transform(z)
        for relation, transform in enumerate(self.relations):
            selected = edge_type == relation
            source, target = edge_index[:, selected]
            if source.numel() == 0:
                continue
            # Gates are not in the denominator: even one sender is gated.
            messages = transform(z[source]) * sender_gate[source, None]
            total = z.new_zeros(z.shape).index_add(0, target, messages)
            count = torch.bincount(target, minlength=z.size(0)).clamp_min(1)
            update = update + total / count[:, None]
        return self.norm(z + self.dropout(F.relu(update)))


class GroupGain(nn.Module):
    """Module A grouping + module B gain prediction and soft group gates.

    Fixed features are independently encoded before max pooling. Grouping is
    deterministic preprocessing, not learned clustering. Generic relations are
    the default; parent-target stance requires an explicit mapping/declaration.
    ``prepare`` can be run once and its GroupGraph results reused each epoch.
    """

    _VARIANTS = {
        'root_only', 'original_graph', 'simple_dedup',
        'feature_only_grouping', 'module_a_only', 'module_b_only',
        'full_model', 'attention_control', 'leave_one_out_only',
    }
    _GAIN_VARIANTS = {
        'full_model', 'module_b_only', 'attention_control', 'leave_one_out_only',
    }

    def __init__(self, in_feats, hid_feats=128, num_classes=2, args=None,
                 device=None, **overrides):
        super().__init__()
        args = SimpleNamespace() if args is None else args
        defaults = {
            'variant': 'full_model',
            'layers': getattr(args, 'n_layers_conv', 2),
            'dropout': getattr(args, 'dropout', 0.2),
            'group_pooling': 'max',
            'group_cos_threshold': 0.95,
            'gate_temperature': 0.2,
            'gate_mode': 'both',
            'lambda_gain': 1.0,
            'candidates_per_graph': 2,
            'contexts_per_candidate': 1,
            'full_context_probability': 0.5,
            'correct_only': True,
            'micro_batch_size': 64,
            'stance_target': 'generic',
            'edge_direction': None,
            'relation_mapping': dict(DEFAULT_RELATION_MAPPING),
            'lr': getattr(args, 'lr', 5e-4),
            'weight_decay': getattr(args, 'weight_decay', 1e-4),
            'seed': getattr(args, 'seed', 0),
        }
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise TypeError('Unknown GroupGain settings: ' + ', '.join(sorted(unknown)))
        self.config = {
            key: overrides.get(key, getattr(args, 'group_gain_' + key, value))
            for key, value in defaults.items()
        }
        self.in_feats, self.hidden_dim = int(in_feats), int(hid_feats)
        self.num_classes = int(num_classes)
        self.max_hop = int(getattr(args, 'max_hop', 1))
        self._validate_config()
        self.relation_mapping = dict(self.config['relation_mapping'])
        self.num_relations = len(self.relation_mapping)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.in_feats, self.hidden_dim), nn.ReLU(),
        )
        self.layers = nn.ModuleList([
            RelationResidualLayer(self.hidden_dim, self.num_relations,
                                  self.config['dropout'])
            for _ in range(self.config['layers'])
        ])
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(5 * self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.group_dropout = nn.Dropout(self.config['dropout'])
        self.classifier = nn.Linear(3 * self.hidden_dim, self.num_classes)
        self._last_diagnostics = {}
        self._last_auxiliary_loss = None
        # Placement follows the project's model.to(device) convention.

    def _validate_config(self):
        c = self.config
        if min(self.in_feats, self.hidden_dim, self.num_classes, self.max_hop) < 1:
            raise ValueError('Feature dimensions, classes and max_hop must be positive')
        if c['variant'] not in self._VARIANTS:
            raise ValueError('Unknown group gain variant')
        if c['group_pooling'] not in {'max', 'mean'}:
            raise ValueError('group_pooling must be max or mean')
        if c['gate_mode'] not in {'both', 'message_only', 'readout_only'}:
            raise ValueError('gate_mode must be both, message_only or readout_only')
        if c['stance_target'] not in {'generic', 'parent', 'source'}:
            raise ValueError('stance_target must be generic, parent or source')
        for key in ('layers', 'candidates_per_graph', 'contexts_per_candidate',
                    'micro_batch_size'):
            if not isinstance(c[key], int) or c[key] < 1:
                raise ValueError(key + ' must be a positive integer')
        for key in ('gate_temperature', 'lr'):
            if not math.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(key + ' must be finite and positive')
        for key in ('lambda_gain', 'weight_decay'):
            if not math.isfinite(c[key]) or c[key] < 0:
                raise ValueError(key + ' must be finite and nonnegative')
        if not 0 <= c['dropout'] < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not 0 <= c['full_context_probability'] <= 1:
            raise ValueError('full_context_probability must be in [0, 1]')
        if not -1 <= c['group_cos_threshold'] <= 1:
            raise ValueError('group_cos_threshold must be in [-1, 1]')
        mapping = c['relation_mapping']
        if (not isinstance(mapping, dict) or 'generic' not in mapping
                or sorted(mapping.values()) != list(range(len(mapping)))):
            raise ValueError('relation_mapping needs generic and unique contiguous IDs')
        if not isinstance(c['correct_only'], bool):
            raise ValueError('correct_only must be a bool')

    def prepare(self, data):
        """Adapt a single Data, PyG Batch, or a list of preprocessed graphs."""
        if isinstance(data, GroupGraph):
            graphs = [data]
        elif isinstance(data, (list, tuple)):
            graphs = []
            for item in data:
                graphs.extend(self.prepare(item))
        elif isinstance(data, Batch):
            graphs = self.prepare(data.to_data_list())
        elif isinstance(data, Data):
            batch = getattr(data, 'batch', None)
            if batch is not None and torch.unique(batch).numel() > 1:
                raise ValueError('Use a PyG Batch for multiple events and correct root offsets')
            mode = {
                'root_only': 'root_only', 'original_graph': 'singleton',
                'module_b_only': 'singleton', 'simple_dedup': 'exact',
                'feature_only_grouping': 'feature_only',
            }.get(self.config['variant'], 'context')
            graphs = [build_group_graph(
                data, threshold=self.config['group_cos_threshold'], mode=mode,
                relation_mapping=self.relation_mapping,
                stance_target=self.config['stance_target'],
                edge_direction=self.config['edge_direction'],
            )]
        else:
            raise TypeError('Expected Data, Batch, GroupGraph or a list of these')
        if not graphs:
            raise ValueError('At least one event graph is required')
        for graph in graphs:
            if graph.x.size(1) != self.in_feats:
                raise ValueError('Node feature dimension does not match the model')
            if graph.relation_mapping != self.relation_mapping:
                raise ValueError('Graph relation mapping does not match the checkpoint')
        return graphs

    def _pool_nodes(self, h, node_to_group, count):
        valid = node_to_group >= 0
        h, mapping = h[valid], node_to_group[valid]
        if self.config['group_pooling'] == 'mean':
            total = h.new_zeros(count, self.hidden_dim).index_add(0, mapping, h)
            sizes = torch.bincount(mapping, minlength=count).clamp_min(1)
            return total / sizes[:, None]
        pooled = h.new_full((count, self.hidden_dim), -torch.inf)
        return pooled.scatter_reduce(
            0, mapping[:, None].expand_as(h), h, reduce='amax', include_self=True,
        )

    def _raw_group_embeddings(self, graph):
        device = self.classifier.weight.device
        h = self.node_encoder(graph.x.to(device=device, dtype=torch.float32))
        return self._pool_nodes(h, graph.node_to_group.to(device), graph.num_groups)

    def _materialize(self, graphs, keep_masks=None):
        if keep_masks is not None and len(keep_masks) != len(graphs):
            raise ValueError('One group mask is required for each event')
        units, edges, types, batches, roots = [], [], [], [], []
        offset = 0
        device = self.classifier.weight.device
        for index, graph in enumerate(graphs):
            keep = (torch.ones(graph.num_groups, dtype=torch.bool, device=device)
                    if keep_masks is None else torch.as_tensor(keep_masks[index], device=device))
            if keep.dtype != torch.bool or keep.shape != (graph.num_groups,):
                raise ValueError('Each mask must be boolean with one entry per group')
            if not keep[0]:
                raise ValueError('Every induced group graph must retain the source')
            # Encode only retained nodes; deletion happens before any GNN work.
            mapping = graph.node_to_group.to(device)
            valid = mapping >= 0
            active_nodes = valid.clone()
            active_nodes[valid] = keep[mapping[valid]]
            remap = torch.full((graph.num_groups,), -1, dtype=torch.long, device=device)
            size = int(keep.sum().item())
            remap[keep] = torch.arange(size, device=device)
            h = self.node_encoder(graph.x.to(device=device, dtype=torch.float32)[active_nodes])
            u = self._pool_nodes(h, remap[mapping[active_nodes]], size)
            edge = graph.edge_index.to(device)
            selected = keep[edge[0]] & keep[edge[1]]
            edge = remap[edge[:, selected]]
            relation = graph.edge_type.to(device)[selected]
            # Inverse relations have distinct IDs. Self transforms are separate.
            edges.append(torch.cat((edge, edge.flip(0)), dim=1) + offset)
            types.append(torch.cat((relation, relation + self.num_relations)))
            units.append(u)
            batches.append(torch.full((size,), index, dtype=torch.long, device=device))
            root = torch.zeros(size, dtype=torch.bool, device=device)
            root[0] = True
            roots.append(root)
            offset += size
        return (torch.cat(units), torch.cat(edges, dim=1), torch.cat(types),
                torch.cat(batches), torch.cat(roots))

    def _predict_full_context(self, u, batch, root_mask, num_graphs):
        replies = ~root_mask
        psi = self.context_encoder(u[replies])
        total = u.new_zeros(num_graphs, self.hidden_dim).index_add(0, batch[replies], psi)
        counts = torch.bincount(batch[replies], minlength=num_graphs)
        denominator = (counts[batch[replies]] - 1).clamp_min(1)[:, None]
        context = (total[batch[replies]] - psi) / denominator
        root = u[root_mask][batch[replies]]
        candidate = u[replies]
        predicted = self.gain_head(torch.cat(
            (root, candidate, context, candidate - context, candidate * context), dim=-1,
        )).flatten()
        values = u.new_zeros(u.size(0)).index_copy(0, replies.nonzero().flatten(), predicted)
        gates = u.new_ones(u.size(0)).index_copy(
            0, replies.nonzero().flatten(),
            torch.sigmoid(predicted / self.config['gate_temperature']),
        )
        return values, gates

    def forward_graphs(self, graphs, keep_masks=None, use_gain=None, return_details=False):
        graphs = self.prepare(graphs)
        u, edge, relation, batch, root_mask = self._materialize(graphs, keep_masks)
        use_gain = self.config['variant'] in self._GAIN_VARIANTS if use_gain is None else use_gain
        if use_gain:
            gains, gates = self._predict_full_context(u, batch, root_mask, len(graphs))
        else:
            gains, gates = u.new_zeros(u.size(0)), u.new_ones(u.size(0))
        message_gate = gates if self.config['gate_mode'] != 'readout_only' else torch.ones_like(gates)
        readout_gate = gates if self.config['gate_mode'] != 'message_only' else torch.ones_like(gates)
        z = self.group_dropout(u)
        for layer in self.layers:
            z = layer(z, edge, relation, message_gate)
        replies = ~root_mask
        weight = readout_gate[replies]
        total = z.new_zeros(len(graphs), self.hidden_dim).index_add(
            0, batch[replies], z[replies] * weight[:, None],
        )
        denominator = z.new_zeros(len(graphs)).index_add(0, batch[replies], weight)
        pooled = total / (denominator[:, None] + 1e-8)
        representation = torch.cat((u[root_mask], z[root_mask], pooled), dim=-1)
        logits = self.classifier(representation)
        details = dict(u=u, z=z, gates=gates, predicted_gains=gains,
                       batch=batch, root_mask=root_mask, edge_index=edge,
                       edge_type=relation, representation=representation)
        self._last_diagnostics = {key: value.detach() for key, value in details.items()}
        self._last_diagnostics['grouping'] = [dict(g.diagnostics) for g in graphs]
        return (logits, details) if return_details else logits

    def forward(self, data):
        return self.forward_graphs(self.prepare(data))

    def gain_predictions(self, graphs, contexts):
        """No label/teacher input; contexts contain only retained group IDs."""
        graphs = self.prepare(graphs)
        if not contexts:
            return self.classifier.weight.new_empty(0)
        embeddings = {}
        encoded = {}
        features = []
        for graph_index, candidate, context in contexts:
            graph = graphs[graph_index]
            if not 1 <= candidate < graph.num_groups:
                raise ValueError('Gain candidates must be non-source groups')
            if (len(set(context)) != len(context) or candidate in context
                    or any(not 1 <= j < graph.num_groups for j in context)):
                raise ValueError('Contexts need distinct other non-source groups')
            if graph_index not in embeddings:
                embeddings[graph_index] = self._raw_group_embeddings(graph)
                encoded[graph_index] = self.context_encoder(embeddings[graph_index])
            u = embeddings[graph_index]
            c = (encoded[graph_index][list(context)].mean(0)
                 if context else u.new_zeros(self.hidden_dim))
            k = u[candidate]
            features.append(torch.cat((u[0], k, c, k - c, k * c)))
        return self.gain_head(torch.stack(features)).flatten()

    def gain_loss(self, graphs, targets):
        if targets.values.ndim != 1 or targets.values.numel() != len(targets.samples):
            raise ValueError('Gain values must match the sampled contexts')
        if not targets.samples:
            return self.classifier.weight.new_zeros(())
        predicted = self.gain_predictions(graphs, targets.samples)
        expected = targets.values.detach().to(predicted)
        self._last_diagnostics['gain_mae'] = (predicted.detach() - expected).abs().mean()
        self._last_diagnostics['gain_targets'] = expected
        self._last_diagnostics['gain_sampling'] = dict(targets.diagnostics)
        return F.smooth_l1_loss(predicted, expected)

    def classification_loss(self, logits, target):
        return F.cross_entropy(logits, target.reshape(-1).long())

    def _labels(self, graphs):
        if any(g.y is None or torch.as_tensor(g.y).numel() != 1 for g in graphs):
            raise ValueError('Training requires one event label per graph')
        return torch.cat([torch.as_tensor(g.y).reshape(1) for g in graphs]).to(
            device=self.classifier.weight.device, dtype=torch.long,
        )

    def compute_loss(self, data, teacher=None, gain_targets=None, lambda_gain=None):
        graphs = self.prepare(data)
        # Call the canonical forward to support the optional EIN wrapper.
        logits = self.forward_graphs(graphs)
        classification = self.classification_loss(logits, self._labels(graphs))
        weight = self.config['lambda_gain'] if lambda_gain is None else lambda_gain
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('lambda_gain must be finite and nonnegative')
        supervised_gain = (self.config['variant'] in self._GAIN_VARIANTS
                           and self.config['variant'] != 'attention_control' and weight > 0)
        if supervised_gain and gain_targets is None:
            if teacher is None:
                raise ValueError('Gain-supervised training requires a frozen teacher or gain_targets')
            gain_targets = compute_gain_targets(
                teacher, graphs,
                candidates_per_graph=self.config['candidates_per_graph'],
                contexts_per_candidate=self.config['contexts_per_candidate'],
                full_context_probability=(1.0 if self.config['variant'] == 'leave_one_out_only'
                                          else self.config['full_context_probability']),
                correct_only=self.config['correct_only'],
                micro_batch_size=self.config['micro_batch_size'],
            )
        gain = self.gain_loss(graphs, gain_targets) if supervised_gain else classification.new_zeros(())
        self._last_auxiliary_loss = weight * gain
        self._last_diagnostics['classification_loss'] = classification.detach()
        self._last_diagnostics['gain_loss'] = gain.detach()
        return classification + self._last_auxiliary_loss

    def teacher_training_loss(self, data, mask_probability=0.5, generator=None):
        if not 0 <= mask_probability <= 1:
            raise ValueError('mask_probability must be in [0, 1]')
        graphs = self.prepare(data)
        masks = []
        for graph in graphs:
            keep = torch.rand(graph.num_groups, generator=generator) >= mask_probability
            keep[0] = True
            masks.append(keep)
        full = self.forward_graphs(graphs, use_gain=False)
        partial = self.forward_graphs(graphs, keep_masks=masks, use_gain=False)
        y = self._labels(graphs)
        return 0.5 * (self.classification_loss(full, y) + self.classification_loss(partial, y))

    def initialize_from_teacher(self, teacher):
        """Copy trained node encoder/backbone/classifier, leave gain head new."""
        if (self.in_feats, self.hidden_dim, self.num_classes, self.relation_mapping,
            len(self.layers), self.config['group_pooling']) != (
                teacher.in_feats, teacher.hidden_dim, teacher.num_classes,
                teacher.relation_mapping, len(teacher.layers), teacher.config['group_pooling']):
            raise ValueError('Teacher and student backbone settings must match')
        for name in ('node_encoder', 'layers', 'classifier'):
            getattr(self, name).load_state_dict(getattr(teacher, name).state_dict())
        return self

    def gain_head_parameters(self):
        return list(self.context_encoder.parameters()) + list(self.gain_head.parameters())

    def physics_loss(self, *unused):
        return self.classifier.weight.new_zeros(())

    def auxiliary_loss(self):
        return (self._last_auxiliary_loss if self._last_auxiliary_loss is not None
                else self.classifier.weight.new_zeros(()))

    def get_diagnostics(self):
        return dict(self._last_diagnostics)

    def init_optimizer(self, args=None):
        return torch.optim.Adam(
            self.parameters(), lr=getattr(args, 'lr', self.config['lr']),
            weight_decay=getattr(args, 'weight_decay', self.config['weight_decay']),
        )

    def save_checkpoint(self, path, metadata=None):
        """Include data label meaning in metadata['label_mapping'] when known."""
        payload = dict(
            format_version=1, state_dict=self.state_dict(), config=dict(self.config),
            in_feats=self.in_feats, hidden_dim=self.hidden_dim,
            num_classes=self.num_classes, max_hop=self.max_hop,
            relation_mapping=dict(self.relation_mapping),
            root_convention='explicit index or unique zero-in-degree source; group zero',
            label_mapping=None, metadata={} if metadata is None else dict(metadata),
        )
        payload['label_mapping'] = payload['metadata'].get('label_mapping')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, map_location='cpu'):
        if not isinstance(map_location, (str, torch.device)):
            raise TypeError('map_location must be a device string or torch.device')
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get('format_version') != 1:
            raise ValueError('Unsupported GroupGain checkpoint version')
        model = cls(payload['in_feats'], payload['hidden_dim'], payload['num_classes'],
                    args=SimpleNamespace(max_hop=payload['max_hop']), **payload['config'])
        model.load_state_dict(payload['state_dict'])
        model.to(map_location)
        model.checkpoint_metadata = payload.get('metadata', {})
        return model


class EINGroupGain(GroupGain):
    """EIN four-output adapter; CE consumes logits through classification_loss.

    For joint training use compute_loss with a frozen teacher in a staged
    training loop. The generic EIN trainer by itself has no teacher lifecycle.
    """

    def forward(self, data):
        logits = self.forward_graphs(self.prepare(data))
        self._last_auxiliary_loss = None
        dummy = logits.new_zeros(logits.size(0), self.max_hop, 1)
        return logits, dummy, dummy, dummy


def freeze_teacher(teacher):
    teacher.eval()
    teacher.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.grad = None
    return teacher


@torch.no_grad()
def compute_gain_targets(teacher, graphs, candidates_per_graph=2,
                         contexts_per_candidate=1, full_context_probability=0.5,
                         correct_only=True, generator=None, micro_batch_size=64,
                         split='train'):
    """CE(Q[S],y)-CE(Q[S+k],y), re-encoding each induced subgraph.

    Only training events/labels may be passed. Micro-batches share original
    feature tensors; they do not allocate K complete copies of an event.
    """
    if split != 'train':
        raise ValueError('Gain targets may only use the training split')
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('The reference teacher must be frozen and in eval mode')
    if any(not isinstance(value, int) or value < 1 for value in
           (candidates_per_graph, contexts_per_candidate, micro_batch_size)):
        raise ValueError('Sampling counts and micro_batch_size must be positive')
    if not 0 <= full_context_probability <= 1:
        raise ValueError('full_context_probability must be in [0, 1]')
    graphs = teacher.prepare(graphs)
    if any(g.diagnostics.get('split', 'train') != 'train' for g in graphs):
        raise ValueError('Gain target input contains a non-training event')
    y = teacher._labels(graphs)
    eligible = torch.ones(len(graphs), dtype=torch.bool, device=y.device)
    full_forwards = 0
    if correct_only:
        predictions = []
        for start in range(0, len(graphs), micro_batch_size):
            predictions.append(teacher.forward_graphs(
                graphs[start:start + micro_batch_size], use_gain=False,
            ).argmax(-1))
            full_forwards += 1
        eligible = torch.cat(predictions) == y
    samples, requests, masks, labels = [], [], [], []
    for index, graph in enumerate(graphs):
        k_count = graph.num_groups - 1
        if not eligible[index] or k_count == 0:
            continue
        candidates = torch.randperm(k_count, generator=generator)[:candidates_per_graph] + 1
        for candidate in candidates.tolist():
            others = [j for j in range(1, graph.num_groups) if j != candidate]
            for _ in range(contexts_per_candidate):
                if torch.rand((), generator=generator).item() < full_context_probability:
                    context = tuple(others)
                else:
                    order = torch.randperm(len(others), generator=generator).tolist()
                    size = int(torch.randint(len(others) + 1, (), generator=generator))
                    context = tuple(sorted(others[j] for j in order[:size]))
                samples.append((index, candidate, context))
                before = torch.zeros(graph.num_groups, dtype=torch.bool)
                before[0] = True
                if context:
                    before[list(context)] = True
                after = before.clone()
                after[candidate] = True
                requests.extend((graph, graph))
                masks.extend((before, after))
                labels.extend((y[index], y[index]))
    losses = []
    for start in range(0, len(requests), micro_batch_size):
        stop = start + micro_batch_size
        logits = teacher.forward_graphs(requests[start:stop], masks[start:stop], use_gain=False)
        losses.append(F.cross_entropy(logits, torch.stack(labels[start:stop]), reduction='none'))
    if losses:
        losses = torch.cat(losses).reshape(-1, 2)
        values = (losses[:, 0] - losses[:, 1]).detach()
    else:
        values = teacher.classifier.weight.new_empty(0)
    diagnostics = dict(
        events=len(graphs), eligible_events=int(eligible.sum().item()),
        eligible_fraction=float(eligible.float().mean().item()),
        samples=len(samples), positive_fraction=float((values > 0).float().mean()) if values.numel() else None,
        negative_fraction=float((values < 0).float().mean()) if values.numel() else None,
        mean_gain=float(values.mean()) if values.numel() else None,
        teacher_forwards=full_forwards + math.ceil(len(requests) / micro_batch_size),
        teacher_full_forwards=full_forwards,
        teacher_masked_forwards=math.ceil(len(requests) / micro_batch_size),
    )
    return GainTargets(samples, values, diagnostics)
