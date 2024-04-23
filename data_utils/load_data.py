import os
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import random_split, DataLoader

from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset
from torch_geometric.utils import degree

from data_utils.wrapper import NewDataset
from utils.process import process_sph


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def fn(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data


def load_data(args):
    if args.dataset in ['NCI1', 'NCI109', 'Mutagenicity', 'PTC_MR', 'AIDS', 'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB',
                        'PROTEINS', 'DD', 'MUTAG', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                        'REDDIT-MULTI-12K']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_tudataset(args)
    elif args.dataset[:4] == 'ogbg':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_ogbg(args)
    elif args.dataset == 'ZINC':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_zinc(args)
    elif args.dataset in ['CLUSTER', 'PATTERN']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_node_cls(args)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=fn)
    val_loader = DataLoader(validation_set, batch_size=args.eval_batch_size, collate_fn=fn)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, collate_fn=fn)
    return train_loader, val_loader, test_loader, num_tasks, num_features, edge_features


def load_tudataset(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    dataset = TUDataset(os.path.join(args.data_root, args.dataset),
                        name=args.dataset,
                        pre_transform=transform
                        )
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    num_tasks = dataset.num_classes
    num_features = dataset.num_features
    num_edge_features = 1
    data = NewDataset(dataset)
    process_sph(args, data)
    num_training = int(len(data) * 0.8)
    num_val = int(len(data) * 0.1)
    num_test = len(data) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(data, [num_training, num_val, num_test])
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_ogbg(args):
    if args.dataset not in ['ogbg-ppa', 'ogbg-code2']:
        transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    else:
        transform = None
    dataset = PygGraphPropPredDataset(name=args.dataset, root=os.path.join(args.data_root, args.dataset),
                                      pre_transform=transform)
    num_tasks = dataset.num_tasks
    num_features = dataset.num_features
    num_edge_features = dataset.num_edge_features
    split_idx = dataset.get_idx_split()
    training_data = dataset[split_idx['train']]
    validation_data = dataset[split_idx['valid']]
    test_data = dataset[split_idx['test']]
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_zinc(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    training_data = ZINC(os.path.join(args.data_root, args.dataset), split='train', subset=True,
                         pre_transform=transform)
    validation_data = ZINC(os.path.join(args.data_root, args.dataset), split='val', subset=True,
                           pre_transform=transform)
    test_data = ZINC(os.path.join(args.data_root, args.dataset), split='test', subset=True,
                     pre_transform=transform)
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')
    num_tasks = 1
    num_features = 28
    num_edge_features = 4
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_node_cls(args):
    transform = T.AddLaplacianEigenvectorPE(k=args.pe_origin_dim, attr_name='pe', is_undirected=True)
    training_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='train',
                                        pre_transform=transform)
    validation_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='val',
                                          pre_transform=transform)
    test_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='test',
                                    pre_transform=transform)
    num_task = training_data.num_classes
    num_feature = training_data.num_features
    num_edge_features = 1
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')

    return num_task, num_feature, num_edge_features, training_set, validation_set, test_set
