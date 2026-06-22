

import os
import json
import torch
import random
import copy
from utils.tools import *
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch.utils.data import Dataset
from torch_geometric.utils import degree, to_networkx, to_undirected
from torch_scatter import scatter



def build_node_features(post, encoder):
    texts = [post['source']['content']]
    texts.extend([comment['content'] for comment in post['comment']])
    return encoder.get_sentence_embeddings(texts)


def build_node_states(post):
    node_state = [0 for _ in range(len(post['comment']) + 1)]
    for comment in post['comment']:
        node_idx = int(comment['comment id']) + 1
        if node_idx < len(node_state):
            node_state[node_idx] = int(comment.get('state', 0))
    return torch.LongTensor(node_state)


def build_edge_stances(post, edge_index):
    edge_stance_map = {}
    for comment in post['comment']:
        if 'stance_label' not in comment:
            continue
        parent = int(comment['parent']) + 1
        child = int(comment['comment id']) + 1
        stance = int(comment['stance_label'])
        edge_stance_map[(parent, child)] = stance
        edge_stance_map[(child, parent)] = stance

    stances = []
    for src, dst in edge_index.t().tolist():
        stances.append(edge_stance_map.get((int(src), int(dst)), -1))
    return torch.LongTensor(stances)


def degree_centrality(data):
    ud_edge_index = to_undirected(data.edge_index)
    centrality = degree(ud_edge_index[1])
    centrality[0] = 1
    return centrality - 1.0 + 1e-8


def pagerank_centrality(data, damp=0.85, k=10):
    device = data.x.device
    bu_edge_index = data.edge_index.clone()
    bu_edge_index[0], bu_edge_index[1] = data.edge_index[1], data.edge_index[0]

    num_nodes = data.num_nodes
    deg_out = degree(bu_edge_index[0])
    centrality = torch.ones((num_nodes,)).to(device).to(torch.float32)

    for _ in range(k):
        edge_msg = centrality[bu_edge_index[0]] / deg_out[bu_edge_index[0]]
        agg_msg = scatter(edge_msg, bu_edge_index[1], reduce='sum')
        pad = torch.zeros((len(centrality) - len(agg_msg),)).to(device).to(torch.float32)
        agg_msg = torch.cat((agg_msg, pad), 0)
        centrality = (1 - damp) * centrality + damp * agg_msg

    centrality[0] = centrality.min().item()
    return centrality


def eigenvector_centrality(data):
    import networkx as nx

    bu_data = data.clone()
    bu_data.edge_index = bu_data.no_root_edge_index
    bu_data.edge_index = to_undirected(bu_data.edge_index)

    graph = to_networkx(bu_data)
    centrality = nx.eigenvector_centrality(graph, tol=1e-3)
    centrality = [centrality[i] for i in range(bu_data.num_nodes)]
    return torch.tensor(centrality, dtype=torch.float32).to(bu_data.x.device)


def betweenness_centrality(data):
    import networkx as nx

    graph = to_networkx(data.clone())
    centrality = nx.betweenness_centrality(graph)
    centrality = [
        centrality[i] if centrality[i] != 0 else centrality[i] + 1e-16
        for i in range(data.num_nodes)
    ]
    return torch.tensor(centrality, dtype=torch.float32).to(data.x.device)


def build_ragcl_centrality(post, x, edge_index, no_root_edge_index, metric):
    metric = metric or 'PageRank'
    centrality_keys = {
        'Degree': 'Degree',
        'PageRank': 'Pagerank',
        'Eigenvector': 'Eigenvector',
        'Betweenness': 'Betweenness',
    }
    if 'centrality' in post and metric in centrality_keys:
        key = centrality_keys[metric]
        if key in post['centrality']:
            return torch.tensor(post['centrality'][key], dtype=torch.float32)

    if x.size(0) <= 1:
        return torch.ones((1,), dtype=torch.float32)

    one_data = Data(
        x=torch.ones(x.size(0), 20),
        edge_index=edge_index,
        no_root_edge_index=no_root_edge_index,
    )
    one_data = Batch.from_data_list([one_data])

    if metric == 'Degree':
        return degree_centrality(one_data)
    if metric == 'PageRank':
        return pagerank_centrality(one_data)
    if metric == 'Eigenvector':
        return eigenvector_centrality(one_data)
    if metric == 'Betweenness':
        return betweenness_centrality(one_data)
    raise ValueError('Unsupported centrality metric: {}'.format(metric))


def attach_optional_fields(data, post):
    if 'domain_id' in post['source']:
        data.domain_id = torch.LongTensor([post['source']['domain_id']])
    return data


class GraphListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    @property
    def num_features(self):
        if len(self.data_list) == 0:
            return 0
        return self.data_list[0].x.size(-1)


def data_label(data):
    return int(data.y.view(-1)[0].item())


def class_counts(dataset):
    counts = {}
    for data in dataset:
        label = data_label(data)
        counts[label] = counts.get(label, 0) + 1
    return counts


def strict_balanced_sample(source_datasets, target_dataset):
    target_counts = class_counts(target_dataset)
    source_by_label = {}
    for dataset in source_datasets:
        for data in dataset:
            label = data_label(data)
            source_by_label.setdefault(label, []).append(data)

    sampled = []
    for label, target_count in sorted(target_counts.items()):
        candidates = source_by_label.get(label, [])
        if len(candidates) < target_count:
            raise ValueError(
                'Strict OOD needs {} source samples for label {}, but only {} are available.'.format(
                    target_count, label, len(candidates)
                )
            )
        sampled.extend(random.sample(candidates, target_count))

    random.shuffle(sampled)
    return GraphListDataset(sampled)


def concat_graph_datasets(datasets):
    data_list = []
    for dataset in datasets:
        data_list.extend([data for data in dataset])
    random.shuffle(data_list)
    return GraphListDataset(data_list)


def post_label(post):
    return post[1]['source']['label']


def post_class_counts(post_list):
    counts = {}
    for post in post_list:
        label = post_label(post)
        counts[label] = counts.get(label, 0) + 1
    return counts


def strict_balanced_sample_posts(source_posts, target_posts):
    target_counts = post_class_counts(target_posts)
    source_by_label = {}
    for post in source_posts:
        label = post_label(post)
        source_by_label.setdefault(label, []).append(post)

    sampled = []
    for label, target_count in sorted(target_counts.items()):
        candidates = source_by_label.get(label, [])
        if len(candidates) < target_count:
            raise ValueError(
                'Strict OOD needs {} source posts for label {}, but only {} are available.'.format(
                    target_count, label, len(candidates)
                )
            )
        sampled.extend(random.sample(candidates, target_count))

    random.shuffle(sampled)
    return sampled


def assign_domain_id(post_list, domain_id):
    assigned = []
    for post_id, post in post_list:
        post = copy.deepcopy(post)
        post['source']['domain_id'] = domain_id
        assigned.append((post_id, post))
    return assigned


def build_split_posts(label_source_path, k_shot=10000, split='622'):
    if split == '622':
        train_split = 0.6
        test_split = 0.8
    elif split == '802':
        train_split = 0.8
        test_split = 0.8
    else:
        raise ValueError('Unsupported split: {}'.format(split))

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))

    random.shuffle(all_post)
    train_post = []

    multi_class = False
    for post in all_post:
        if post[1]['source']['label'] == 2 or post[1]['source']['label'] == 3:
            multi_class = True

    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0
    for post in all_post[:int(len(all_post) * train_split)]:
        if post[1]['source']['label'] == 0 and num0 != k_shot:
            train_post.append(post)
            num0 += 1
        if post[1]['source']['label'] == 1 and num1 != k_shot:
            train_post.append(post)
            num1 += 1
        if post[1]['source']['label'] == 2 and num2 != k_shot:
            train_post.append(post)
            num2 += 1
        if post[1]['source']['label'] == 3 and num3 != k_shot:
            train_post.append(post)
            num3 += 1
        if multi_class:
            if num0 == k_shot and num1 == k_shot and num2 == k_shot and num3 == k_shot:
                break
        else:
            if num0 == k_shot and num1 == k_shot:
                break

    if split == '622':
        val_post = all_post[int(len(all_post) * train_split):int(len(all_post) * test_split)]
        test_post = all_post[int(len(all_post) * test_split):]
    elif split == '802':
        val_post = all_post[-1:]
        test_post = all_post[int(len(all_post) * test_split):]

    return train_post, val_post, test_post


def write_split_posts(label_dataset_path, train_post, val_post, test_post):
    train_path, val_path, test_path = dataset_makedirs(label_dataset_path)
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)
    return (
        os.path.join(label_dataset_path, 'train'),
        os.path.join(label_dataset_path, 'val'),
        os.path.join(label_dataset_path, 'test')
    )



class ResGCNTreeDataset(InMemoryDataset):
    def __init__(self, root, word_embedding, word2vec, undirected, transform=None, pre_transform=None,
                 pre_filter=None, args=None):
        self.word_embedding = word_embedding
        self.word2vec = word2vec
        self.args = args
        self.undirected = undirected
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        for filename in raw_file_names:
            y = []
            row = []
            col = []
            no_root_row = []
            no_root_col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            x = build_node_features(post, self.word2vec)
            node_state = build_node_states(post)
            
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            for i, comment in enumerate(post['comment']):
                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            edge_index = [row, col]
            no_root_edge_index = [no_root_row, no_root_col]
            y = torch.LongTensor(y)
            edge_index = torch.LongTensor(edge_index)
            no_root_edge_index = torch.LongTensor(no_root_edge_index)
            centrality = build_ragcl_centrality(
                post,
                x,
                edge_index,
                no_root_edge_index,
                getattr(self.args, 'centrality', 'PageRank'),
            )
            edge_index = to_undirected(edge_index) if self.undirected else edge_index
            edge_stance = build_edge_stances(post, edge_index)

            state = post['state']

            # Populate the matrix with state counts from each hop
            user_state = []
            for hop in sorted(state.keys(), key=lambda x: int(x.split('-')[0])):
                row = [0, state[hop]['state_0'], state[hop]['state_1']]
                user_state.append(row)

            num_hop = len(user_state)

            # padding
            for i in range(self.args.max_hop-len(user_state)):
                row = [0, 0, 0]
                user_state.append(row)

            # Convert the list to a PyTorch tensor for matrix format
            user_state = torch.tensor(user_state, dtype=torch.float32)
            user_state = user_state.unsqueeze(0) # to (1, n, 2)
            

            one_data = Data(x=x, y=y, edge_index=edge_index, no_root_edge_index=no_root_edge_index, user_state=user_state, num_hop=num_hop, node_state=node_state, edge_stance=edge_stance, centrality=centrality) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index, no_root_edge_index=no_root_edge_index, user_state=user_state, num_hop=num_hop, node_state=node_state, edge_stance=edge_stance, centrality=centrality)
            one_data = attach_optional_fields(one_data, post)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

class TreeDataset(InMemoryDataset):
    def __init__(self, root, word_embedding, word2vec, transform=None, pre_transform=None,
                 pre_filter=None, args=None):
        self.word_embedding = word_embedding
        self.args = args
        self.word2vec = word2vec
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        for filename in raw_file_names:
            y = []
            row = []
            col = []
            no_root_row = []
            no_root_col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            x = build_node_features(post, self.word2vec)
            node_state = build_node_states(post)
            
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            for i, comment in enumerate(post['comment']):
                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            edge_index = [row, col]
            no_root_edge_index = [no_root_row, no_root_col]
            y = torch.LongTensor(y)
            edge_index = torch.LongTensor(edge_index)
            no_root_edge_index = torch.LongTensor(no_root_edge_index)
            centrality = build_ragcl_centrality(
                post,
                x,
                edge_index,
                no_root_edge_index,
                getattr(self.args, 'centrality', 'PageRank'),
            )
            edge_stance = build_edge_stances(post, edge_index)
            
            state = post['state']

            # Populate the matrix with state counts from each hop
            user_state = []
            for hop in sorted(state.keys(), key=lambda x: int(x.split('-')[0])):
                row = [0, state[hop]['state_0'], state[hop]['state_1']]
                user_state.append(row)

            num_hop = len(user_state)

            # padding
            for i in range(self.args.max_hop-len(user_state)):
                row = [0, 0, 0]
                user_state.append(row)

            # Convert the list to a PyTorch tensor for matrix format
            user_state = torch.tensor(user_state, dtype=torch.float32)
            user_state = user_state.unsqueeze(0) # to (1, n, 2)

            one_data = Data(x=x, y=y, edge_index=edge_index, no_root_edge_index=no_root_edge_index, user_state=user_state, num_hop=num_hop, node_state=node_state, edge_stance=edge_stance, centrality=centrality) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index, no_root_edge_index=no_root_edge_index, user_state=user_state, num_hop=num_hop, node_state=node_state, edge_stance=edge_stance, centrality=centrality)
            one_data = attach_optional_fields(one_data, post)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

def split_dataset(label_source_path, label_dataset_path, k_shot=10000, split='622'):
    print('Spliting data...')
    train_post, val_post, test_post = build_split_posts(label_source_path, k_shot, split)
    write_split_posts(label_dataset_path, train_post, val_post, test_post)
    print('Finished.')
