import argparse
import torch
import numpy as np
import random
import torch_geometric.transforms as T
from model import *
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from hetero_data_utils import *
import scipy.sparse as sp


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Confidence-based Graph MoE")
    parser.add_argument('--setting', type=str, default='exp',
                        help='decide framework settings')
    parser.add_argument('--method', type=str, default='vanilla', choices=["vanilla", "SimpleGate", "SimpleGateInTurn"])
    parser.add_argument('--submethod', type=str, default='none',
                        choices=["none", "pretrain_model1", "pretrain_model2", "pretrain_both"])
    parser.add_argument('--subloss', type=str, default='loss_combine', choices=["combine_loss", "loss_combine"])
    parser.add_argument('--infer_method', type=str, default='simple', choices=["simple", "multi"])
    parser.add_argument('--dataset', type=str, default='flickr',
                        choices=['flickr', 'cora', 'citeseer', 'pubmed', 'arxiv',
                                 'penn94', 'genius', 'twitch-gamer', 'ogbn-proteins',
                                 'snap-patents', 'arxiv-year', 'pokec', "product"])
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--no_preset_struc', default=False, action='store_true')
    parser.add_argument('--no_cached', default=False, action='store_true')
    parser.add_argument('--no_add_self_loops', default=False, action='store_true')
    parser.add_argument('--no_normalize', default=False, action='store_true')
    parser.add_argument('--no_save_all_time', default=False, action='store_true')

    parser.add_argument('--model1', type=str, default='MLP',
                        )
    parser.add_argument('--model2', type=str, default='GCN', choices=["GCN","GAT",
                                                                      "Sage","gcn-spmoe",
                                                                      "sage-spmoe","GPR-GNN",
                                                                      "adagcn","MLP","GIN"]
                        )
    parser.add_argument('--model1_hidden_dim', type=int, default=64)
    parser.add_argument('--model2_hidden_dim', type=int, default=64)
    parser.add_argument('--model1_num_layers', type=int, default=2)
    parser.add_argument('--model2_num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument("--hyper_hidden", default=64, type=int)
    parser.add_argument("--hyper_num_layers", default=2, type=int)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_gate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropout_gate', type=float, default=0.5)
    parser.add_argument('--crit', type=str, default='nllloss', choices=['nllloss', 'bceloss', 'crossentropy'])
    # parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--confidence', type=str, default='variance')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--no_batch_norm', default=False, action='store_true')

    parser.add_argument('--epoch', type=int, default=20000)
    parser.add_argument('--big_epoch', type=int, default=20000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--big_patience', type=int, default=10)
    parser.add_argument('--m_times', type=int, default=50)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--coef", type=float, default=1)
    parser.add_argument("--gpr_alpha", type=float, default=0.1)
    parser.add_argument("--adagcn_layers", type=int, default=2)
    parser.add_argument("--adagcn_max", type=int, default=500)

    parser.add_argument("--early_signal", type=str, default="val_loss", choices=["val_loss", "val_acc"])


    parser.add_argument('--cpu', default=False, action="store_true")
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--print_freq', type=int, default=100)

    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--log', default=False, action='store_true')

    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--split_run_idx', type=int, default=0)

    parser.add_argument('--model_save_path', type=str, default='saved_model4/')
    parser.add_argument("--denoise_save_path", type=str, default='denoise_data1/')
    parser.add_argument('--wandb_save_path', type=str, default='new_imp/')
    parser.add_argument('--biased', type=str, default='none', choices=['none', 'dispersion', 'logit'])
    parser.add_argument('--original_data', type=str, choices=["true","false","hypermlp","hypergcn","none"], default="none")
    parser.add_argument('--grid_range', default=False, action="store_true")
    parser.add_argument("--no_early_stop", default=False, action="store_true")


    return parser.parse_args(args)

class ArgNote(object):
    def __init__(self, setting="exp", method="vanilla", submethod="none", subloss="loss_combine", infer_method="simple",
                 dataset="flickr", seed=2023, no_preset_struc=False, no_cached=False, no_add_self_loops=False, no_normalize=False,
                 no_save_all_time=False, model1="MLP", model2="Sage", model1_hidden_dim=64, model2_hidden_dim=64,
                 model1_num_layers=2, model2_num_layers=2, heads=1, lr=0.001, lr_gate=0.001, weight_decay=0.0005, dropout=0.5,
                 dropout_gate=0.5, crit="nllloss", adam=False, activation="relu", alpha=1, confidence="variance",
                 train_prop=0.5, valid_prop=0.25, no_batch_norm=False, epoch=20000, big_epoch=20000, patience=100, big_patience=10,
                 m_times=50, cpu=False, print_freq=100, run=10, log=False, rand_split=False, directed=False, split_run_idx=0,
                 model_save_path="saved_model4/", wandb_save_path="new_imp/", biased="none", original_data="false", grid_range=False,
                 adagcn_layers=5, early_signal="val_loss",num_experts=2, no_early_stop=False, max=500):
        self.setting = setting
        self.method = method
        self.submethod = submethod
        self.subloss = subloss
        self.infer_method = infer_method
        self.dataset = dataset
        self.seed = seed
        self.no_preset_struc = no_preset_struc
        self.no_cached = no_cached
        self.no_add_self_loops = no_add_self_loops
        self.no_save_all_time = no_save_all_time
        self.no_normalize = no_normalize
        self.model1 = model1
        self.model2 = model2
        self.model1_hidden_dim = model1_hidden_dim
        self.model2_hidden_dim = model2_hidden_dim
        self.model1_num_layers = model1_num_layers
        self.model2_num_layers = model2_num_layers
        self.heads = heads
        self.lr = lr
        self.lr_gate = lr_gate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.dropout_gate = dropout_gate
        self.crit = crit
        self.adam = adam
        self.activation = activation
        self.alpha = alpha
        self.confidence = confidence
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.no_batch_norm = no_batch_norm
        self.epoch = epoch
        self.big_epoch = big_epoch
        self.patience = patience
        self.num_experts = num_experts
        self.no_early_stop = no_early_stop
        self.big_patience = big_patience
        self.m_times = m_times
        self.cpu = cpu
        self.print_freq = print_freq
        self.run = run
        self.log = log
        self.rand_split = rand_split
        self.directed = directed
        self.split_run_idx = split_run_idx
        self.model_save_path = model_save_path
        self.wandb_save_path = wandb_save_path
        self.biased = biased
        self.original_data = original_data
        self.grid_range = grid_range
        self.adagcn_layers = adagcn_layers
        self.early_signal = early_signal
        self.max=max

def verify_args(args):
    if args.subloss == 'combine_loss':
        if args.infer_method != 'simple':
            raise ValueError('Invalid Args Combination. Combine_loss has to be used with Simple infer method.')
    if args.dataset in ['penn94', 'twitch-gamer', 'arxiv-year', 'pokec', 'snap-patents', 'genius', 'ogbn-proteins']:
        args.setting = 'hetero_' + args.setting
        args.no_cached = True


def print_info(args):
    s = ' | lr ({}) dp ({}) lr_gate ({}) dp_gate ({}) alpha ({}) method ({}) sub-method ({}) subloss ({}) infer-method ({}) model1_hidden ({}) model1_layer ({}) model2_hidden ({}) model2_layer ({}) no_cached ({}) no_save_all_time ({}) no_batch_norm ({}) use_adam ({}) biased ({}) original_data ({}) confidence_no_grad'.format(
        args.lr, args.dropout, args.lr_gate, args.dropout_gate, args.alpha, args.method, args.submethod, args.subloss,
        args.infer_method, args.model1_hidden_dim, args.model1_num_layers, args.model2_hidden_dim,
        args.model2_num_layers, args.no_cached, args.no_save_all_time, args.no_batch_norm, args.adam, args.biased,
        args.original_data)
    return s


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_preset_struc(args):
    dataset = args.dataset
    if dataset == 'flickr':
        args.model1_hidden_dim = 256
        args.model2_hidden_dim = 256
        args.model1_num_layers = 3
        args.model2_num_layers = 3
        if args.model2 == "GAT":
            args.model1_num_layers = 2
            args.model2_num_layers = 2
            args.model1_hidden_dim = 12
            args.model2_hidden_dim = 12
        elif args.model2 == "Sage":
            args.model1_num_layers = 3
            args.model2_num_layers = 3
            args.model1_hidden_dim = 256
            args.model2_hidden_dim = 256
        # args.lr = 0.01
    elif dataset in ['arxiv', "product"]:
        args.model1_hidden_dim = 256
        args.model2_hidden_dim = 256
        args.model1_num_layers = 3
        args.model2_num_layers = 3
        args.weight_decay = 0
        if dataset == "arxiv" and args.model2 == "GAT":
            args.model1_num_layers = 2
            args.model2_num_layers = 2
            args.model1_hidden_dim = 32
            args.model2_hidden_dim = 32
        elif dataset == "arxiv" and args.model2 == "Sage":
            args.model1_hidden_dim = 512
            args.model2_hidden_dim = 512
            args.model1_num_layers = 4
            args.model2_num_layers = 4

        if dataset == "product" and args.model2 == "GCN":
            args.no_batch_norm = True
            args.no_cached = True
        elif dataset == "product" and args.model2 == "GAT":
            args.model1_num_layers = 2
            args.model2_num_layers = 2
            # args.no_batch_norm = True
            # args.no_cached = True
            # args.no_normalize = True
        elif dataset == "product" and args.model2 == "Sage":
            args.model1_hidden_dim = 256
            args.model2_hidden_dim = 256
            args.model1_num_layers = 3
            args.model2_num_layers = 3
            args.no_batch_norm = True


    elif dataset in ['penn94', 'twitch-gamer', 'arxiv-year', 'pokec', 'snap-patents', 'genius', 'ogbn-proteins']:
        args.model1_hidden_dim = 64
        args.model2_hidden_dim = 64
        args.weight_decay = 0.001
        if dataset in ['penn94']:
            args.no_cached = True
            if args.model2 == "GAT":
                args.model1_hidden_dim = 12
                args.model2_hidden_dim = 12
            elif args.model2 == "Sage":
                args.model1_hidden_dim = 32
                args.model2_hidden_dim = 32
        elif dataset in ['arxiv-year']:
            if args.model2 == "GAT":
                args.model1_hidden_dim = 32
                args.model2_hidden_dim = 32
            elif args.model2 == "Sage":
                args.model1_hidden_dim = 32
                args.model2_hidden_dim = 32
        elif dataset in ['pokec']:
            if args.model2 == "GAT":
                args.model1_hidden_dim = 12
                args.model2_hidden_dim = 12
            elif args.model2 == "Sage":
                args.model1_hidden_dim = 12
                args.model2_hidden_dim = 12
        elif dataset in ['twitch-gamer']:
            if args.model2 == "GAT":
                args.model1_hidden_dim = 8
                args.model2_hidden_dim = 8
            elif args.model2 == "Sage":
                args.model1_hidden_dim = 32
                args.model2_hidden_dim = 32

        # args.lr = 0.1
    elif dataset in ['cora', 'pubmed', 'citeseer']:
        args.no_cached = True
        args.no_batch_norm = True
        args.model1_hidden_dim = 16
        args.model2_hidden_dim = 16
        args.adam = True
        if args.model2 == "GAT":
            args.model2_hidden_dim = 8
            args.model1_hidden_dim = 8
            if dataset == "pubmed":
                args.weight_decay = 0.001

        # if dataset == 'cora':
        #     args.lr = 0.01
        # elif dataset == 'citeseer':
        #     args.lr = 0.1
        # elif dataset == 'pubmed':
        #     args.lr = 0.1

    if args.model2 == "gcn-spmoe":
        args.model1_num_layers = 0
        args.model2_num_layers = 0
    return args


def process_ogbn_arxiv(data, train_idx, valid_idx, test_idx):
    n = data.num_nodes
    train_mask = create_mask(n, train_idx)
    val_mask = create_mask(n, valid_idx)
    test_mask = create_mask(n, test_idx)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def create_mask(n, pos):
    res = torch.zeros(n, dtype=torch.bool)
    res[pos] = True
    return res


def mask2idx(data):
    split_idx = {'train': (data.train_mask == True).nonzero(as_tuple=True)[0],
                 'valid': (data.val_mask == True).nonzero(as_tuple=True)[0],
                 'test': (data.test_mask == True).nonzero(as_tuple=True)[0]}
    return split_idx


def load_data(args):
    if args.dataset == 'flickr':

        from torch_geometric.datasets.flickr import Flickr
        if args.model2 in ["GPR-GNN", "adagcn"]:
            dataset = Flickr(root='/tmp/Flickr')
        else:
            dataset = Flickr(root='/tmp/Flickr', transform=T.ToSparseTensor())
        data = dataset[0]
        data.y = data.y.view(-1, 1)
        data.mlp_x = data.x
        split_idx = mask2idx(data)
    elif args.dataset == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        if args.model2 != "GPR-GNN" and args.model2 != "adagcn":
            data = T.ToSparseTensor()(data)
        data.mlp_x = data.x
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data = process_ogbn_arxiv(data, train_idx, valid_idx, test_idx)
    elif args.dataset == 'product':
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected
        if args.model2 in ["GPR-GNN", "adagcn"]:
            dataset = PygNodePropPredDataset(name='ogbn-products')
        else:
            dataset = PygNodePropPredDataset(name='ogbn-products',
                                         transform=T.ToSparseTensor())
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # adj_t = data.adj_t.set_diag()
        # deg = adj_t.sum(dim=1).to(torch.float)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        # data.adj_t = adj_t
        data.mlp_x = data.x
        data = process_ogbn_arxiv(data, train_idx, valid_idx, test_idx)
    elif args.setting in ['hetero_exp', 'hetero_ten'] and args.dataset in ['penn94', 'twitch-gamer', 'arxiv-year',
                                                                           'pokec', 'snap-patents', 'genius',
                                                                           'ogbn-proteins']:
        dataset, data, split_idx = load_hetero_data(args)
        data.mlp_x = data.x
    elif args.dataset in ['cora', 'citeseer', 'pubmed']:
        from torch_geometric.datasets.planetoid import Planetoid
        if args.dataset == 'cora':
            dataset = Planetoid(root='/tmp/Planetoid', name='Cora', transform=T.NormalizeFeatures())
        elif args.dataset == 'citeseer':
            dataset = Planetoid(root='/tmp/Planetoid', name='Citeseer', transform=T.NormalizeFeatures())
        elif args.dataset == 'pubmed':
            dataset = Planetoid(root='/tmp/Planetoid', name='Pubmed', transform=T.NormalizeFeatures())
        data = dataset[0]
        if args.model2 != "GPR-GNN" and args.model2 != "adagcn":
            data = T.ToSparseTensor()(data)
        data.y = data.y.view(-1, 1)
        data.mlp_x = data.x
        split_idx = mask2idx(data)
    else:
        raise ValueError('Invalid dataset name or wrong combination i.e., homophily vs heterophily')
    return dataset, data, split_idx


def get_model_hyperparameters(args, data, dataset):
    if args.method == 'vanilla':  # for flickr
        input_dim = data.num_features
        hidden_dim = args.model2_hidden_dim
        if not args.setting in ['hetero_exp', 'hetero_ten']:
            output_dim = dataset.num_classes
        else:
            output_dim = data.num_classes
        num_layers = args.model2_num_layers
        return input_dim, hidden_dim, output_dim, num_layers
    elif args.method in ['MoWSE', 'SimpleGate', 'SimpleGateInTurn']:
        input_dim = data.num_features
        hidden_dim2 = args.model2_hidden_dim
        if not args.setting in ['hetero_exp', 'hetero_ten']:
            output_dim = dataset.num_classes
        else:
            output_dim = data.num_classes
        num_layers2 = args.model2_num_layers
        hidden_dim1 = args.model1_hidden_dim
        num_layers1 = args.model1_num_layers
        return [input_dim, hidden_dim1, output_dim, num_layers1], [input_dim, hidden_dim2, output_dim, num_layers2]


def load_model(model_name, input_dim, hidden_dim, output_dim, num_layers, dropout, args):
    if model_name == 'GCN':
        return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'GAT':
        return GAT(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'Sage':
        return Sage(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'GIN':
        return GIN(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'MLP':
        return MLP(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'MLPLearn':
        return MLPLearn(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == "gcn-spmoe":
        return GCN_SpMoE(input_dim, hidden_dim, output_dim,
                          num_layers, dropout,
                          num_experts=args.num_experts, k=args.k, coef=args.coef)
    elif model_name == "sage-spmoe":
        return SAGE_SpMoE(input_dim, hidden_dim, output_dim,
                          num_layers, dropout,
                          num_experts=args.num_experts, k=args.k, coef=args.coef)
    elif model_name == "GPR-GNN":
        return GPRGNN(input_dim, hidden_dim, output_dim, Init='PPR', dprate=.0, dropout=dropout, K=10, alpha=args.gpr_alpha,
                 Gamma=None, num_layers=3)
    elif model_name == "adagcn":
        return AdaGCN(nfeat=input_dim,  nhid=hidden_dim, nclass=output_dim, dropout=dropout, dropout_adj=dropout)


def save_model(args, model, label):
    s = args.model_save_path
    s += args.setting + '_'
    s += args.method + '_'
    s += args.submethod + '_'
    s += args.subloss + '_'
    s += args.infer_method + '_'
    s += args.dataset + '_'
    s += args.model1 + "_"
    s += str(args.model1_hidden_dim) + "_"
    s += str(args.model1_num_layers) + "_"
    s += args.model2 + "_"
    s += str(args.model2_hidden_dim) + "_"
    s += str(args.model2_num_layers) + "_"
    s += str(args.heads) + "_"
    s += str(args.lr) + '_'
    s += str(args.dropout) + '_'
    s += str(args.lr_gate) + '_'
    s += str(args.dropout_gate) + '_'
    s += str(args.weight_decay) + '_'
    s += str(args.no_batch_norm) + '_'
    s += str(args.no_cached) + '_'
    s += str(args.no_add_self_loops) + '_'
    s += str(args.no_normalize) + '_'
    s += str(args.no_save_all_time) + '_'
    s += args.confidence + '_'
    s += str(args.biased) + '_'
    s += args.original_data + '_'
    s += args.early_signal + '_'
    s += str(args.num_experts) + "_"
    s += str(args.no_early_stop) + '_'
    s += str(args.adagcn_layers) + "_"
    s += str(args.alpha) + '_' + label + '.pt'
    torch.save(model.state_dict(), s)
    return s


def save_moe_model(args, model, model_id):
    s = args.model_save_path
    s += args.setting + '_'
    s += args.method + '_'
    s += args.submethod + '_'
    s += args.subloss + '_'
    s += args.infer_method + '_'
    s += args.dataset + '_'
    s += args.model1 + "_"
    s += str(args.model1_hidden_dim) + "_"
    s += str(args.model1_num_layers) + "_"
    s += args.model2 + "_"
    s += str(args.model2_hidden_dim) + "_"
    s += str(args.model2_num_layers) + "_"
    s += str(args.heads) + "_"
    s += str(args.lr) + '_'
    s += str(args.dropout) + '_'
    s += str(args.lr_gate) + '_'
    s += str(args.dropout_gate) + '_'
    s += str(args.weight_decay) + '_'
    s += str(args.no_batch_norm) + '_'
    s += str(args.no_cached) + '_'
    s += str(args.no_add_self_loops) + '_'
    s += str(args.no_normalize) + '_'
    s += str(args.no_save_all_time) + '_'
    s += args.confidence + '_'
    s += str(args.biased) + '_'
    s += args.original_data + '_'
    s += args.early_signal + '_'
    s += str(args.num_experts) + "_"
    s += str(args.no_early_stop) + '_'
    s += str(args.adagcn_layers) + "_"
    s += str(args.alpha) + '_{}.pt'.format(model_id)
    torch.save(model.state_dict(), s)
    return s


def append_results(RES, res):
    train_list, valid_list, test_list, weight_list, val_loss_list = RES
    train_score, valid_score, test_score, weight_score, val_loss = res
    train_list.append(train_score)
    valid_list.append(valid_score)
    test_list.append(test_score)
    weight_list.append(weight_score)
    val_loss_list.append(-val_loss)


def create_result_dict(RES, args):
    train_list, valid_list, test_list, weight_list, val_loss_list = RES
    res = {}
    res["train score (avg)"] = np.mean(train_list)
    res["train score (std)"] = np.std(train_list)
    res["valid score (avg)"] = np.mean(valid_list)
    res["valid score (std)"] = np.std(valid_list)
    res["test score (avg)"] = np.mean(test_list)
    res["test score (std)"] = np.std(test_list)
    res["weight (avg)"] = np.mean(weight_list)
    res["weight (std)"] = np.std(weight_list)

    res["valloss (avg)"] = np.mean(val_loss_list)
    res["valloss (std)"] = np.std(val_loss_list)


    res["train score (all)"] = train_list
    res["train score (max)"] = np.max(train_list)
    res["train score (min)"] = np.min(train_list)

    res["valid score (all)"] = train_list
    res["valid score (max)"] = np.max(valid_list)
    res["valid score (min)"] = np.min(valid_list)

    res["test score (all)"] = train_list
    res["test score (max)"] = np.max(test_list)
    res["test score (min)"] = np.min(test_list)

    res["valloss (all)"] = val_loss_list
    res["vallosse (max)"] = np.max(val_loss_list)
    res["valloss (min)"] = np.min(val_loss_list)

    res["weight (all)"] = weight_list



    res["dataset"] = args.dataset
    res["setting"] = args.setting
    res["method"] = args.method
    res["submethod"] = args.submethod
    res["subloss"] = args.subloss
    res["infer_method"] = args.infer_method
    res["seed"] = args.seed
    res["no_preset_struc"] = args.no_preset_struc
    res["no_cached"] = args.no_cached
    res["no_add_self_loops"] = args.no_add_self_loops
    res["no_normalize"] = args.no_normalize
    res["no_save_all_time"] = args.no_save_all_time
    res["model1"] = args.model1
    res["model2"] = args.model2
    res["model1_hidden_dim"] = args.model1_hidden_dim
    res["model2_hidden_dim"] = args.model2_hidden_dim
    res["model1_num_layers"] = args.model1_num_layers
    res["model2_num_layers"] = args.model2_num_layers
    res["heads"] = args.heads

    res["early_signal"] = args.early_signal

    res["lr"] = args.lr
    res["lr_gate"] = args.lr_gate
    res["weight_decay"] = args.weight_decay
    res["dropout"] = args.dropout
    res["dropout_gate"] = args.dropout_gate
    res["crit"] = args.crit
    res["adam"] = args.adam
    res["activation"] = args.activation
    res["alpha"] = args.alpha
    res["confidence"] = args.confidence
    res["train_prop"] = args.train_prop
    res["valid_prop"] = args.valid_prop
    res["no_batch_norm"] = args.no_batch_norm
    res["epoch"] = args.epoch
    res["big_epoch"] = args.big_epoch
    res["patience"] = args.patience
    res["big_patience"] = args.big_patience
    res["m_times"] = args.m_times

    res["cpu"] = args.cpu
    res["print_freq"] = args.print_freq
    res["run"] = args.run
    res["log"] = args.log
    res["rand_split"] = args.rand_split
    res["directed"] = args.directed
    res["split_run_idx"] = args.split_run_idx
    res["model_save_path"] = args.model_save_path
    res["wandb_save_path"] = args.wandb_save_path
    res["biased"] = args.biased
    res["original_data"] = args.original_data
    res["grid_range"] = args.grid_range
    res["num_experts"] = args.num_experts
    res["k"] = args.k
    res["coef"] = args.coef
    res["gpr_alpha"] = args.gpr_alpha
    res["adagcn_layers"] = args.adagcn_layers
    res["adagcn_max"] = args.adagcn_max

    return res


def grid_combo(dataset):
    if dataset in ["arxiv", "product"]:
        lrs_large = [0.01]
        dps_large = [0.5]
        lrs = [0.01, 0.001, 0.1]
        dps = [0.5]
    elif dataset == "arxiv-year":
        lrs_large = [0.001, 0.01, 0.1]
        dps_large = [0.5, 0.1, 0.2, 0.3, 0.4]
        lrs = [0.1, 0.01, 0.001]
        dps = [0.5, 0.2, 0.3, 0.4, 0.1]
    else:
        lrs_large = [0.1, 0.01, 0.001]
        dps_large = [0.1, 0.2, 0.3, 0.4, 0.5]
        lrs = [0.1, 0.01, 0.001]
        dps = [0.1, 0.2, 0.3, 0.4, 0.5]

    return lrs_large, dps_large, lrs, dps
