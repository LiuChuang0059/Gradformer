import configargparse


def set_config(args):
    if "hiv" in args.dataset:
        args.criterion = 'ogbg'
        args.task = 'binary_class'
        args.node_method = 'ogb'
        args.edge_method = 'ogb'
        args.mpnn = 'GCN'
        args.pool = 'mean'
        args.projection = 'mlp'
    elif args.dataset == 'ZINC':
        args.criterion = 'mae'
        args.task = 'regression'
        args.node_method = 'embedding'
        args.edge_method = 'embedding'
        args.mpnn = 'GINE'
        args.pool = 'add'
        args.projection = 'mlp'
    elif args.dataset == 'CLUSTER':
        args.criterion = 'accuracy'
        args.task = 'multi_class'
        args.node_method = 'linear'
        args.edge_method = 'embedding'
        args.mpnn = 'GCN'
        args.pool = 'None'
        args.projection = 'mlp'
    elif args.dataset == 'PATTERN':
        args.criterion = 'accuracy_sbm'
        args.task = 'multi_weight_class'
        args.node_method = 'no_pe'
        args.edge_method = 'embedding'
        args.mpnn = 'GCN'
        args.pool = 'None'
        args.projection = 'mlp'
    else:
        args.criterion = 'accuracy'
        args.task = 'multi_class'
        args.node_method = 'add'
        args.edge_method = 'None'
        args.mpnn = 'GIN'
        args.pool = 'add'
        args.projection = 'lin'


def gene_arg():
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                           description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)
    parser.add_argument('--wandb_run_idx', type=str, default=None)

    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag|augment]')

    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='maximum sequence length to predict (default: None)')

    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    group.add_argument('--gamma', type=float, default=0.5)
    group.add_argument('--learnable', type=bool, default=True)
    group.add_argument('--pool', type=str, default='mean')
    group.add_argument('--task', type=str, default='multi_class')
    group.add_argument('--criterion', type=str, default='accuracy')
    group = parser.add_argument_group('gnn')
    group.add_argument('--node_method', type=str, default='linear')
    group.add_argument('--edge_method', type=str, default='None')
    group.add_argument('--mpnn', type=str, default='gcn')
    group.add_argument('--projection', type=str, default='mlp')
    group.add_argument('--gnn_virtual_node', action='store_true')
    group.add_argument('--dropout', type=float, default=0)
    group.add_argument('--drop_prob', type=float, default=0.0)
    group.add_argument('--attn_dropout', type=float, default=0.5)
    group.add_argument('--gnn_num_layer', type=int, default=5,
                       help='number of GNN message passing layers (default: 5)')
    group.add_argument('--channels', type=int, default=64,
                       help='dimensionality of hidden units in GNNs (default: 64)')
    group.add_argument('--gnn_JK', type=str, default='last')
    group.add_argument('--gnn_residual', action='store_true', default=False)
    group.add_argument('--num_layers', type=int, default=10,
                       help='number of GNN message passing layers (default: 5)')
    group.add_argument('--nhead', type=int, default=4,
                       help='number of GNN message passing layers (default: 5)')

    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=int, default=0,
                       help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=32,
                       help='input batch size for training (default: 128)')
    group.add_argument('--eval_batch_size', type=int, default=32,
                       help='input batch size for training (default: train batch size)')
    group.add_argument('--epochs', type=int, default=100,
                       help='number of epochs to train (default: 100)')
    group.add_argument('--warmup_epoch', type=int, default=5)
    group.add_argument('--num_workers', type=int, default=0,
                       help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=str, default=None)
    group.add_argument('--pct_start', type=float, default=0.3)
    group.add_argument('--weight_decay', type=float, default=1e-5)
    group.add_argument('--grad_clip', type=float, default=None)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--max_lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=1)
    group.add_argument('--start-eval', type=int, default=15)
    group.add_argument('--resume', type=str, default=None)
    group.add_argument('--seed', type=int, default=12344)
    group.add_argument('--run', type=int, default=10)
    group.add_argument('--n_hop', type=int, default=5)
    group.add_argument('--slope', type=float, default=0.0)
    group.add_argument('--pe_norm', type=bool, default=False)
    group.add_argument('--pe_origin_dim', type=int, default=20)
    group.add_argument('--pe_dim', type=int, default=20)

    args, _ = parser.parse_known_args()

    return args
