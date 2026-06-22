import torch
import torch_geometric.utils as tg_utils
from torch_geometric.data.batch import Batch


def graph_centrality(data):
    if not hasattr(data, 'centrality') or data.centrality is None:
        raise ValueError(
            'RAGCL augmentation requires a centrality field. '
            'Regenerate the dataset cache with a RAGCL config.'
        )
    return data.centrality


def degree_drop_weights(data, aggr='mean', norm=True):
    centrality = graph_centrality(data)
    w_row = centrality[data.edge_index[0]].to(torch.float32)
    w_col = centrality[data.edge_index[1]].to(torch.float32)
    s_row = torch.log(w_row) if norm else w_row
    s_col = torch.log(w_col) if norm else w_col
    if aggr == 'sink':
        scores = s_col
    elif aggr == 'source':
        scores = s_row
    else:
        scores = (s_col + s_row) * 0.5
    denom = scores.max() - scores.mean()
    if torch.isclose(denom, torch.zeros_like(denom)):
        return torch.ones_like(scores)
    return (scores.max() - scores) / denom


def drop_edge_weighted(data, edge_weights, p, threshold):
    denom = edge_weights.mean().clamp_min(1e-12)
    edge_weights = edge_weights / denom * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    if sel_mask.sum() == 0 and data.edge_index.size(1) > 0:
        sel_mask[0] = True
    return data.edge_index[:, sel_mask]


def node_aug_weights(centrality, norm=True):
    scores = torch.log(centrality) if norm else centrality
    denom = scores.max() - scores.mean()
    if torch.isclose(denom, torch.zeros_like(denom)):
        return torch.ones_like(scores)
    return (scores.max() - scores) / denom


def aug_node_weighted(node_weights, p, threshold):
    denom = node_weights.mean().clamp_min(1e-12)
    node_weights = node_weights / denom * p
    node_weights = node_weights.where(
        node_weights < threshold, torch.ones_like(node_weights) * threshold
    )
    return torch.bernoulli(1.0 - node_weights).to(torch.bool)


def drop_edge(batch_data, aggr, p, threshold):
    aug_data = batch_data.clone()
    aug_data_list = aug_data.to_data_list()
    for i in range(aug_data.num_graphs):
        if aug_data_list[i].num_nodes > 1 and aug_data_list[i].edge_index.size(1) > 0:
            edge_weights = degree_drop_weights(aug_data_list[i], aggr=aggr)
            aug_edge_index = drop_edge_weighted(aug_data_list[i], edge_weights, p, threshold)
            aug_data_list[i].edge_index = aug_edge_index
    return Batch.from_data_list(aug_data_list).to(aug_data.x.device)


def drop_node(batch_data, p, threshold):
    aug_data = batch_data.clone()
    aug_data_list = aug_data.to_data_list()
    for i in range(aug_data.num_graphs):
        centrality = graph_centrality(aug_data_list[i])
        node_weights = node_aug_weights(centrality)
        sel_mask = aug_node_weighted(node_weights, p, threshold)
        sel_mask[0] = True
        if sel_mask.sum() == 0:
            sel_mask[0] = True
        aug_edge_index, _ = tg_utils.subgraph(
            sel_mask,
            aug_data_list[i].edge_index,
            relabel_nodes=True,
            num_nodes=aug_data_list[i].num_nodes,
        )
        aug_data_list[i].x = aug_data_list[i].x[sel_mask]
        aug_data_list[i].edge_index = aug_edge_index
        aug_data_list[i].num_nodes = aug_data_list[i].x.shape[0]
    return Batch.from_data_list(aug_data_list).to(aug_data.x.device)


def mask_attr(batch_data, p, threshold):
    aug_data = batch_data.clone()
    aug_data_list = aug_data.to_data_list()
    for i in range(aug_data.num_graphs):
        centrality = graph_centrality(aug_data_list[i])
        node_weights = node_aug_weights(centrality)
        sel_mask = aug_node_weighted(node_weights, p, threshold)
        sel_mask[0] = True
        mask_token = torch.zeros_like(aug_data_list[i].x[0], dtype=torch.float)
        aug_data_list[i].x[sel_mask] = mask_token
    return Batch.from_data_list(aug_data_list).to(aug_data.x.device)


def augment(batch_data, augs):
    aug_data = batch_data
    for aug in augs:
        aug_args = aug.split(',')
        if aug_args[0] == 'DropEdge':
            aug_data = drop_edge(aug_data, aug_args[1], float(aug_args[2]), float(aug_args[3]))
        elif aug_args[0] == 'NodeDrop':
            aug_data = drop_node(aug_data, float(aug_args[1]), float(aug_args[2]))
        elif aug_args[0] == 'AttrMask':
            aug_data = mask_attr(aug_data, float(aug_args[1]), float(aug_args[2]))
        else:
            raise ValueError('Unsupported RAGCL augmentation: {}'.format(aug_args[0]))
    return aug_data
