from utils import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score
from torch.distributions import Categorical
import pandas as pd


def get_engine(args, device, data, dataset, split_idx):
    if args.method == 'vanilla':
        engine = BaselineEngine(args, device, data, dataset, split_idx)
    elif args.method == 'MoWSE':
        engine = MoWSEEngine(args, device, data, dataset, split_idx)
    elif args.method == 'SimpleGate':
        engine = SimpleGate(args, device, data, dataset, split_idx)
    elif args.method == 'SimpleGateInTurn':
        engine = SimpleGateInTurn(args, device, data, dataset, split_idx)

    return engine


class Evaluator(object):
    def __init__(self, name):
        self.name = name
        if self.name in ['flickr', 'arxiv', "product", 'penn94', 'pokec', 'arxiv-year', 'snap-patents', 'twitch-gamer',
                         'cora', 'citeseer', 'pubmed']:
            self.eval_metric = 'acc'
        elif self.name in ['ogbn-proteins', 'genius']:
            self.eval_metric = 'rocauc'

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc' or self.eval_metric == 'acc_ccp':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
        return y_true, y_pred

    def eval(self, input_dict):
        if self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc_ccp':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc_ccp(y_true, y_pred)

    def _eval_acc(self, y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(correct.float().sum().item() / len(correct))

        return sum(acc_list) / len(acc_list)

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''
        rocauc_list = []
        y_true = y_true.detach().cpu().numpy()
        if y_true.shape[1] == 1:
            # use the predicted class for single-class classification
            y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
        else:
            y_pred = y_pred.detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

                rocauc_list.append(score)

        if len(rocauc_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute ROC-AUC.')

        return sum(rocauc_list) / len(rocauc_list)


def train_vanilla(model, data, crit, optimizer, args):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    if args.dataset in ['ogbn-proteins', 'genius']:
        if data.y.shape[1] == 1:
            true_label = F.one_hot(data.y, data.y.max() + 1).squeeze(1)
        else:
            true_label = data.y
        loss = crit(out[data.train_mask], true_label.squeeze(1)[data.train_mask].to(torch.float))
    else:
        loss = crit(F.log_softmax(out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_vanilla(model, data, split_idx, evaluator, args):
    model.eval()
    out = model(data)
    if args.dataset in ['ogbn-proteins', 'genius']:
        y_pred = out
    else:
        y_pred = out.argmax(dim=-1, keepdim=True)


    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_acc, valid_acc, test_acc, 0


@torch.no_grad()
def val_loss_vanilla(model, data, crit, args):
    model.eval()
    out = model(data)

    if args.dataset in ['ogbn-proteins', 'genius']:
        if data.y.shape[1] == 1:
            true_label = F.one_hot(data.y, data.y.max() + 1).squeeze(1)
        else:
            true_label = data.y
        loss = crit(out[data.train_mask], true_label.squeeze(1)[data.train_mask].to(torch.float))
    else:
        loss = crit(F.log_softmax(out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])

    return loss.item()


def vanilla_train_test_wrapper(args, model, data, crit, optimizer, split_idx, evaluator, additional=None):
    check = 0
    best_score = 0
    best_loss = float('-inf')

    for i in range(args.epoch):
        loss = train_vanilla(model, data, crit, optimizer, args)
        val_loss = val_loss_vanilla(model, data, crit, args)
        val_loss = - val_loss
        result = test_vanilla(model, data, split_idx, evaluator, args)

        if i % args.print_freq == 0:
            print('{} epochs trained, loss {:.4f}'.format(i, loss))

        if args.early_signal == "val_loss":
            # if result[1] > best_score:
            if val_loss > best_loss:
                check = 0
                best_loss = val_loss
                best_score = result[1]
                if additional:
                    saved_model = save_model(args, model, f'only_model{additional}')
                else:
                    saved_model = save_model(args, model, 'only_model')
            else:
                check += 1
                if check > args.patience:
                    print("{} epochs trained, best val loss {:.4f}".format(i, -best_loss))
                    break
        else:
            if result[1] > best_score:
                check = 0
                best_loss = val_loss
                best_score = result[1]
                if additional:
                    saved_model = save_model(args, model, f'only_model{additional}')
                else:
                    saved_model = save_model(args, model, 'only_model')
            else:
                check += 1
                if check > args.patience:
                    print("{} epochs trained, best val acc {:.4f}".format(i, best_score))
                    break

    if (args.early_signal == "val_loss" and best_loss > float('-inf')) or (args.early_signal == "val_acc" and best_score > 0):
        model.load_state_dict(torch.load(saved_model))
    result = test_vanilla(model, data, split_idx, evaluator, args)
    train_acc, val_acc, test_acc, _ = result
    print('Final results: Train {:.2f} Val {:.2f} Test {:.2f}'.format(train_acc * 100,
                                                                      val_acc * 100,
                                                                      test_acc * 100))
    return train_acc, val_acc, test_acc, _, best_loss

from torch.utils.data import TensorDataset, DataLoader
def get_dataloaders(idx, labels, batch_size=None):
    # labels = torch.LongTensor(labels_np.astype(np.int32))
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                   for phase, dataset in datasets.items()}
    return dataloaders

from typing import List, Tuple, Dict
import copy


def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])

def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx

def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(exclude_idx(idx, [train_idx]), nstopping, replace=False)
    return train_idx, stopping_idx

def gen_splits(
        labels: np.ndarray, idx_split_args: Dict[str, int],
        test: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(known_idx, labels[known_idx], **stopping_split_args)
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx

import scipy.sparse as sp

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes) # self-loop
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec) # ()**(-1/2)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr) # vec to matrix
    return D_invsqrt_corr @ A @ D_invsqrt_corr

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices.astype(np.float32)),
            torch.FloatTensor(coo.data),
            coo.shape)



import time
import logging


from enum import Enum, auto
class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()

class Best(Enum):
    RANKED = auto()
    ALL = auto()
import operator
from torch.nn import Module
class EarlyStopping:
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 10, max_epochs: int = 200, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le) # <
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge) # >
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int) -> bool:
        checks = [self.comp_ops[i](val, self.best_vals[i]) for i, val in enumerate(values)] # [acc > best, loss < best]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [self.comp_ops[i](val, self.remembered_vals[i]) for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values) # record the best performance
                    self.best_state = {key: value.cpu() for key, value in self.model.state_dict().items()} # record the best parameters of network
            elif self.remember is Best.RANKED:
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            self.best_state = {key: value.cpu() for key, value in self.model.state_dict().items()}
                            break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0

import scipy.sparse.linalg as spla
def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm


from torch_geometric.utils import to_scipy_sparse_matrix
def vanilla_ori_adagcn_wrapper(args, model, dataset, data, optimizer, split_idx, device):
    model_args = {
        'hiddenunits': [64],
        'drop_prob': args.dropout,
        'propagation': None,  # propagation involves sparse Tensor\
        'lr': args.lr ,
        'hid_AdaGCN': args.model2_hidden_dim,
        'layers': args.adagcn_layers,
        'dropoutadj_GCN': args.dropout,
        'dropoutadj_AdaGCN': args.dropout,
        'weight_decay': args.weight_decay,
        "max":args.adagcn_max
    }
    stopping_args = dict(stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS], patience=100, max_epochs=10000,
                         remember=Best.RANKED)
    stopping_args['max_epochs'] = args.adagcn_max
    stopping_args['patience'] = args.patience
    model_reg = 5e-3


    test = True
    if not args.setting in ['hetero_exp', 'hetero_ten']:
        nclasses = dataset.num_classes
    else:
        nclasses = data.num_classes

    features = normalize_attributes(data.x).to(device)

    x = to_scipy_sparse_matrix(data.edge_index)
    x = calc_A_hat(x)
    adj = sparse_matrix_to_torch(x).to(device)

    sample_weights = torch.ones(adj.shape[0])
    sample_weights = sample_weights[split_idx['train']]
    sample_weights = sample_weights / sample_weights.sum()
    sample_weights = sample_weights.to(device)

    results = torch.zeros(adj.shape[0], nclasses).to(device)
    labels_all = data.y.view(-1)

    logging.log(21, f"{AdaGCN.__name__}: {model_args}")

    dataloaders = get_dataloaders(split_idx, labels_all)
    early_stopping = EarlyStopping(model, **stopping_args)

    epoch_stats = {'train': {}, 'valid': {}}
    start_time = time.time()
    last_time = start_time

    ALL_epochs = 0

    for layer in range(args.adagcn_layers):
        logging.info(f"|This is the {layer+1}th layer!")
        print(f"|This is the {layer + 1}th layer!")


        for epoch in range(early_stopping.max_epochs):  # 10000
            for phase in epoch_stats.keys():  # 2 phases: train, stopping, train 1 epoch, evaluate on stopping dataset
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0
                running_corrects = 0

                for idx, labels in dataloaders[phase]:  # training set / early stopping set
                    idx = idx.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):  # train: True
                        log_preds = model(features, adj)[idx]  # A is in model's buffer
                        loss = F.nll_loss(log_preds, labels, reduction='none')  # each loss
                        # core 1: weighted loss
                        if phase == 'train':
                            loss = loss * sample_weights
                        preds = torch.argmax(log_preds, dim=1)
                        loss = loss.sum()
                        l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                        loss = loss + model_reg / 2 * l2_reg  # cross loss + L2 regularization
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # Collect statistics
                        running_loss += loss.item()
                        # running_loss += loss.item() * idx.size(0)
                        running_corrects += torch.sum(preds == labels)

                # Collect statistics (current epoch)
                epoch_stats[phase]['loss'] = running_loss / len(dataloaders[phase].dataset)
                epoch_stats[phase]['acc'] = running_corrects.item() / len(dataloaders[phase].dataset)

            # print logging each interval
            if epoch % args.print_freq == 0:
                duration = time.time() - last_time  # each interval including training and early-stopping
                last_time = time.time()

                logging.info(f"Epoch {epoch}: "
                             f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                             f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                             f"early stopping loss = {epoch_stats['valid']['loss']:.2f}, "
                             f"early stopping acc = {epoch_stats['valid']['acc'] * 100:.1f} "
                             f"({duration:.3f} sec)")

                print(f"Epoch {epoch}: "
                             f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                             f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                             f"early stopping loss = {epoch_stats['valid']['loss']:.2f}, "
                             f"early stopping acc = {epoch_stats['valid']['acc'] * 100:.1f} "
                             f"({duration:.3f} sec)")

            # (4) check whether it stops on some epoch
            if len(early_stopping.stop_vars) > 0:
                stop_vars = [epoch_stats['valid'][key] for key in early_stopping.stop_vars] # 'acc', 'loss'
                if early_stopping.check(stop_vars, epoch): # whether exist improvement for patience times
                    break


        # (6) SAMME.R
        ALL_epochs += epoch
        runtime = time.time() - start_time
        logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")
        print(f"Last epoch: {epoch}, ({runtime:.3f} sec)")
        # Load best model weights

        # model.load_state_dict(torch.load(saved_model))
        model.load_state_dict(early_stopping.best_state)
        model.eval()
        # output = model(features, adj)[torch.arange(graph.adj_matrix.shape[0])].detach()
        output = model(features, adj)[torch.arange(adj.shape[0])].detach()
        output_logp = torch.log(F.softmax(output, dim=1))
        h = (nclasses - 1) * (output_logp - torch.mean(output_logp, dim=1).view(-1, 1))
        results += h
        # adjust weights
        # temp = F.nll_loss(output_logp[idx_all['train']],
        #                   torch.LongTensor(labels_all[idx_all['train']].astype(np.int32)).to(device),
        #                   reduction='none')  # 140*1
        temp = F.nll_loss(output_logp[split_idx['train']],
                          torch.LongTensor(labels_all[split_idx['train']]).to(device),
                          reduction='none')  # 140*1
        weight = sample_weights * torch.exp((1 - (nclasses - 1)) / (nclasses - 1) * temp)  # update weights
        weight = weight / weight.sum()
        sample_weights = weight.detach()

        # update features
        features = SparseMM.apply(adj, features).detach()  # adj: tensor[2810, 2810],  features: tensor[2810,2879]

    # (5) evaluate the best model from early stopping on test set
    runtime = time.time() - start_time
    # stopping_preds = torch.argmax(results[idx_all['stopping']], dim=1).cpu().numpy()
    # stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()

    train_preds = torch.argmax(results[split_idx['train']], dim=1).cpu()
    train_acc = (train_preds == labels_all[split_idx['train']]).float().mean()

    stopping_preds = torch.argmax(results[split_idx['valid']], dim=1).cpu()
    stopping_acc = (stopping_preds == labels_all[split_idx['valid']]).float().mean()
    logging.log(21, f"Early stopping accuracy: {stopping_acc * 100:.1f}%")
    print(f"Early stopping accuracy: {stopping_acc * 100:.1f}%")

    # valtest_preds = torch.argmax(results[idx_all['valtest']], dim=1).cpu().numpy()
    # valtest_acc = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    valtest_preds = torch.argmax(results[split_idx['test']], dim=1).cpu()
    valtest_acc = (valtest_preds == labels_all[split_idx['test']]).float().mean()

    valtest_name = 'Test' if test else 'Validation'
    print(f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")
    logging.log(22, f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")

    # (6) return result
    result = {}
    result['early_stopping'] = {'accuracy': stopping_acc}
    result['valtest'] = {'accuracy': valtest_acc}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime / (ALL_epochs + 1)

    train_acc = float(train_acc)
    valid_acc = float(stopping_acc)
    test_acc = float(valtest_acc)



    print('Final results: Train {:.2f} Val {:.2f} Test {:.2f}'.format(train_acc * 100,
                                                                      valid_acc * 100,
                                                                      test_acc * 100))





    return train_acc, valid_acc, test_acc, float(0), float(0)

# deprecated
# def compute_confidence(args, x):
#     n = x.shape[1]
#     if args.confidence == 'variance':
#         out = nn.Softmax(1)(x)
#         variance = torch.var(out, dim=1, unbiased=False)
#         zero_tensor = torch.zeros(n)
#         zero_tensor[0] = 1
#         max_variance = torch.var(zero_tensor, unbiased=False)
#         # var = (out - 1 / n) ** 2 / n
#         # conf = (var.sum(1) * (n ** 2 / (n - 1)))
#     return variance / max_variance


# @torch.no_grad()
def compute_confidence(logit, method):
    n_classes = logit.shape[1]
    logit = nn.Softmax(1)(logit)
    if method == "variance":
        # logit = torch.exp(logit)
        variance = torch.var(logit, dim=1, unbiased=False)
        zero_tensor = torch.zeros(n_classes)
        zero_tensor[0] = 1
        max_variance = torch.var(zero_tensor, unbiased=False)

        res = variance / max_variance
    elif method == "entropy":
        res = 1 - Categorical(probs=logit).entropy() / np.log(n_classes)

    return res.view(-1, 1)


def mowse_train_test_wrapper(args, model1, model2, data, crit, optimizer1, optimizer2, split_idx, evaluator, device):
    big_epoch_check = 0
    big_best_score = 0

    for j in range(args.big_epoch):
        print('------ Big epoch {} Model 1 ------'.format(j))
        model1_turn_val_acc = mowse_train_test_model1_turn_wrapper(args, model1, model2, data, crit, optimizer1,
                                                                   split_idx, evaluator, device, big_best_score)
        print('------ Big epoch {} Model 2 ------'.format(j))
        mowse_train_test_model2_turn_wrapper(args, model1, model2, data, crit, optimizer2, split_idx, evaluator, device,
                                             model1_turn_val_acc)
        result = test_mowse(model1, model2, data, split_idx, evaluator, args, device, False)

        if args.log:
            wandb.log({'Train Acc': result[0],
                       'Val Acc': result[1],
                       'Test Acc': result[2],
                       'Confidence': get_cur_confidence(args, model1, data).mean().item()})
        if result[1] > big_best_score:
            big_epoch_check = 0
            big_best_score = result[1]
            saved_model1_big = save_moe_model(args, model1, '1_big')
            saved_model2_big = save_moe_model(args, model2, '2_big')
        else:
            big_epoch_check += 1
            if big_epoch_check > args.big_patience:
                print("{} big epochs trained, best val accuracy {:.4f}".format(j, big_best_score))
                break

    model1.load_state_dict(torch.load(saved_model1_big))
    model2.load_state_dict(torch.load(saved_model2_big))
    result = test_mowse(model1, model2, data, split_idx, evaluator, args, device, True)
    train_acc, val_acc, test_acc = result
    print('{} big epochs trained, Train {:.2f} Val {:.2f} Test {:.2f}'.format(j, train_acc * 100,
                                                                              val_acc * 100,
                                                                              test_acc * 100))
    return result


def mowse_train_test_model1_turn_wrapper(args, model1, model2, data, crit, optimizer1, split_idx, evaluator, device,
                                         big_best_score):
    if not args.no_save_all_time:
        saved_model1_previous_big_turn = save_moe_model(args, model1, '1_big')
        saved_model2_previous_big_turn = save_moe_model(args, model2, '2_big')

    check = 0
    best_score = 0
    for i in range(args.epoch):

        loss = train_mowse1(model1, model2, data, crit, optimizer1, args)
        result = test_mowse(model1, model2, data, split_idx, evaluator, args, device, False)

        if i % args.print_freq == 0:
            print('{} epochs trained, loss {:.4f}'.format(i, loss))

        if result[1] > best_score:
            check = 0
            best_score = result[1]
            saved_model1 = save_moe_model(args, model1, '1_inner')
            saved_model2 = save_moe_model(args, model2, '2_inner')
        else:
            check += 1
            if check > args.patience:
                print("{} epochs trained, best val accuracy {:.2f}".format(i, best_score * 100))
                break
    if not args.no_save_all_time:
        if best_score > big_best_score:
            model1.load_state_dict(torch.load(saved_model1))
            model2.load_state_dict(torch.load(saved_model2))
        else:
            model1.load_state_dict(torch.load(saved_model1_previous_big_turn))
            model2.load_state_dict(torch.load(saved_model2_previous_big_turn))
        saved_model1 = save_moe_model(args, model1, 1)
        saved_model2 = save_moe_model(args, model2, 2)
        return max(best_score, big_best_score)
    else:
        model1.load_state_dict(torch.load(saved_model1))
        model2.load_state_dict(torch.load(saved_model2))
        return best_score


def mowse_train_test_model2_turn_wrapper(args, model1, model2, data, crit, optimizer2, split_idx, evaluator, device,
                                         model1_turn_val_acc):
    if not args.no_save_all_time:
        saved_model1_previous_small_turn = save_moe_model(args, model1, 1)
        saved_model2_previous_small_turn = save_moe_model(args, model2, 2)

    check = 0
    best_score = 0
    for i in range(args.epoch):

        loss = train_mowse2(model1, model2, data, crit, optimizer2, args)
        result = test_mowse(model1, model2, data, split_idx, evaluator, args, device, False)

        if i % args.print_freq == 0:
            print('{} epochs trained, loss {:.4f}'.format(i, loss))

        if result[1] > best_score:
            check = 0
            best_score = result[1]
            saved_model1 = save_moe_model(args, model1, '1_inner')
            saved_model2 = save_moe_model(args, model2, '2_inner')
        else:
            check += 1
            if check > args.patience:
                print("{} epochs trained, best val accuracy {:.2f}".format(i, best_score * 100))
                break
    if not args.no_save_all_time:
        if best_score > model1_turn_val_acc:
            model1.load_state_dict(torch.load(saved_model1))
            model2.load_state_dict(torch.load(saved_model2))
        else:
            model1.load_state_dict(torch.load(saved_model1_previous_small_turn))
            model2.load_state_dict(torch.load(saved_model2_previous_small_turn))
        saved_model1 = save_moe_model(args, model1, 1)
        saved_model2 = save_moe_model(args, model2, 2)
    else:
        model1.load_state_dict(torch.load(saved_model1))
        model2.load_state_dict(torch.load(saved_model2))


def train_mowse1(model1, model2, data, crit, optimizer1, args):
    model1.train()
    model2.eval()
    optimizer1.zero_grad()
    if args.confidence == 'variance':
        model1_x = model1(data)[data.train_mask]
        model1_conf = compute_confidence(args, model1_x)
    elif args.confidence == 'learnable':
        model1_x, model1_conf = model1(data)
        model1_x = model1_x[data.train_mask]
        model1_conf = model1_conf[data.train_mask]
    model1_conf = (model1_conf ** args.alpha).view(-1)
    # print(model1_conf)
    # print(model1_x)
    with torch.no_grad():
        model2_x = model2(data)[data.train_mask]
    if crit.__class__.__name__ == 'BCEWithLogitsLoss':
        loss_model1 = crit(F.softmax(model1_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
        loss_model2 = crit(F.softmax(model2_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
    else:
        loss_model1 = crit(F.log_softmax(model1_x, dim=1), data.y.squeeze(1)[data.train_mask])
        loss_model2 = crit(F.log_softmax(model2_x, dim=1), data.y.squeeze(1)[data.train_mask])

    loss = model1_conf * loss_model1 + (1 - model1_conf) * loss_model2
    loss.mean().backward()
    optimizer1.step()
    return loss.mean().item()


def train_mowse2(model1, model2, data, crit, optimizer2, args):
    model1.eval()
    model2.train()
    optimizer2.zero_grad()
    model2_x = model2(data)[data.train_mask]

    with torch.no_grad():
        if args.confidence == 'variance':
            model1_x = model1(data)[data.train_mask]
            model1_conf = compute_confidence(args, model1_x)
        elif args.confidence == 'learnable':
            model1_x, model1_conf = model1(data)
            model1_x = model1_x[data.train_mask]
            model1_conf = model1_conf[data.train_mask]

    model1_conf = (model1_conf ** args.alpha).view(-1)

    if crit.__class__.__name__ == 'BCEWithLogitsLoss':
        loss_model1 = crit(F.softmax(model1_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
        loss_model2 = crit(F.softmax(model2_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
    else:
        loss_model1 = crit(F.log_softmax(model1_x, dim=1), data.y.squeeze(1)[data.train_mask])
        loss_model2 = crit(F.log_softmax(model2_x, dim=1), data.y.squeeze(1)[data.train_mask])

    loss = model1_conf * loss_model1 + (1 - model1_conf) * loss_model2
    loss.mean().backward()
    optimizer2.step()
    return loss.mean().item()


@torch.no_grad()
def test_mowse(model1, model2, data, split_idx, evaluator, args, device, check_g_dist):
    model1.eval()
    model2.eval()
    if args.confidence == 'variance':
        model1_x = model1(data)
        model1_conf = compute_confidence(args, model1_x)
    elif args.confidence == 'learnable':
        model1_x, model1_conf = model1(data)

    model1_conf = (model1_conf ** args.alpha).view(-1)
    model1_out = nn.Softmax(1)(model1_x)
    model2_out = nn.Softmax(1)(model2(data))

    tmp_train_acc_list = []
    tmp_val_acc_list = []
    tmp_test_acc_list = []

    for t in range(args.m_times):
        m = torch.rand(model1_conf.shape).to(device)
        gate = (m < model1_conf).int().view(-1, 1)
        model1_pred = model1_out.argmax(dim=-1, keepdim=True)
        model2_pred = model2_out.argmax(dim=-1, keepdim=True)
        y_pred = model1_pred.view(-1) * gate.view(-1) + model2_pred.view(-1) * (1 - gate.view(-1))
        y_pred = y_pred.view(-1, 1)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })
        tmp_train_acc_list.append(train_acc)
        tmp_val_acc_list.append(valid_acc)
        tmp_test_acc_list.append(test_acc)

    train_acc = np.mean(tmp_train_acc_list)
    valid_acc = np.mean(tmp_val_acc_list)
    test_acc = np.mean(tmp_test_acc_list)

    if check_g_dist:
        gate = gate.view(-1)
        model1_percent = gate.sum().item() / len(gate) * 100
        print('Model1 / Model2: {:.2f} / {:.2f}'.format(model1_percent, 100 - model1_percent))

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def get_cur_confidence(args, model1, data):
    model1.eval()
    if args.confidence == 'variance':
        model1_x = model1(data)[data.train_mask]
        model1_conf = compute_confidence(args, model1_x) ** args.alpha
    elif args.confidence == 'learnable':
        model1_x, model1_conf = model1(data)
        model1_conf = (model1_conf[data.train_mask] ** args.alpha).view(-1)
    return model1_conf


class BaselineEngine(object):
    def __init__(self, args, device, data, dataset, split_idx):
        self.args = args
        self.device = device
        if args.model2 != "adagcn":
            self.data = data.to(self.device)
        else:
            self.data = data
        self.dataset = dataset
        input_dim, hidden_dim, output_dim, num_layers = get_model_hyperparameters(args, data, dataset)
        model_name = args.model2
        dropout = args.dropout
        self.model = load_model(model_name, input_dim, hidden_dim, output_dim, num_layers, dropout, args).to(
            self.device)
        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)
        if self.args.dataset in ['genius', 'ogbn-genius']:
            self.crit = nn.BCEWithLogitsLoss()
        else:
            self.crit = nn.NLLLoss()
        if self.args.adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr,
                weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model.reset_parameters()
        if self.args.model2 == "adagcn":

            result = vanilla_ori_adagcn_wrapper(self.args, self.model, self.dataset,
                                                self.data, self.optimizer, self.split_idx, self.device)
        else:

            result = vanilla_train_test_wrapper(self.args, self.model, self.data, self.crit, self.optimizer, self.split_idx,
                                                self.evaluator)
        return result


class MoWSEEngine(object):
    def __init__(self, args, device, data, dataset, split_idx):
        self.args = args
        self.device = device
        self.data = data.to(self.device)
        model1_hyper, model2_hyper = get_model_hyperparameters(args, data, dataset)
        input_dim, hidden_dim1, output_dim, num_layers1 = model1_hyper
        input_dim, hidden_dim2, output_dim, num_layers2 = model2_hyper
        dropout = args.dropout
        if args.confidence == 'learnable':
            self.model1 = load_model(args.model1 + 'Learn', input_dim, hidden_dim1, output_dim, num_layers1,
                                     dropout).to(self.device)
        else:
            self.model1 = load_model(args.model1, input_dim, hidden_dim1, output_dim, num_layers1, dropout).to(
                self.device)
        self.model2 = load_model(args.model2, input_dim, hidden_dim2, output_dim, num_layers2, dropout).to(self.device)

        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)
        if self.args.dataset in ['genius', 'ogbn-genius']:
            self.crit = nn.BCEWithLogitsLoss(reduction='none')
            self.crit_pretrain = nn.BCEWithLogitsLoss()
        else:
            self.crit = nn.NLLLoss(reduction='none')
            self.crit_pretrain = nn.NLLLoss()

        if self.args.optim == 'adam':
            self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.args.lr,
                                               weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.args.lr,
                                               weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model1.reset_parameters()
        self.model2.reset_parameters()

        if self.args.pretrain == True:
            result1 = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                                 self.optimizer1, self.split_idx, self.evaluator)
            result2 = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                                 self.optimizer2, self.split_idx, self.evaluator)
            print('Model pretraining done')

        result = mowse_train_test_wrapper(self.args, self.model1, self.model2, self.data, self.crit,
                                          self.optimizer1, self.optimizer2,
                                          self.split_idx, self.evaluator, self.device)

        return result1, result2, result


class SimpleGate(object):
    def __init__(self, args, device, data, dataset, split_idx):
        self.args = args
        self.device = device
        self.data = data.to(self.device)
        model1_hyper, model2_hyper = get_model_hyperparameters(args, data, dataset)
        input_dim, hidden_dim1, output_dim, num_layers1 = model1_hyper
        input_dim, hidden_dim2, output_dim, num_layers2 = model2_hyper
        dropout = args.dropout

        self.model1 = load_model(args.model1, input_dim, hidden_dim1, output_dim, num_layers1, dropout, args).to(
            self.device)
        self.model2 = load_model(args.model2, input_dim, hidden_dim2, output_dim, num_layers2, dropout, args).to(
            self.device)

        if args.biased == "logit":
            if args.original_data == "true":
                self.gate_model = GateMLP(input_dim + output_dim, hidden_dim1, 1, num_layers1, dropout, args).to(
                    self.device)
            elif args.original_data == "false":
                self.gate_model = GateMLP(output_dim, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        elif args.biased == "dispersion":
            if args.original_data == "true":
                self.gate_model = GateMLP(input_dim + 2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
            elif args.original_data == "false":
                self.gate_model = GateMLP(2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
            elif args.original_data == "hypermlp":
                self.para1_model = GateMLP(in_channels=input_dim,
                                           hidden_channels=args.hyper_hidden,
                                           out_channels=2 * args.model1_hidden_dim,
                                           num_layers=args.hyper_num_layers,
                                           dropout=0.5, args=args).to(self.device)
                self.parabias1_model = GateMLP(in_channels=input_dim,
                                               hidden_channels=args.hyper_hidden,
                                               out_channels=args.model1_hidden_dim,
                                               num_layers=args.hyper_num_layers,
                                               dropout=0.5, args=args).to(self.device)
                self.para2_model = GateMLP(in_channels=input_dim,
                                           hidden_channels=args.hyper_hidden,
                                           out_channels=args.model1_hidden_dim,
                                           num_layers=args.hyper_num_layers,
                                           dropout=0.5, args=args).to(self.device)
                self.parabias2_model = GateMLP(in_channels=input_dim,
                                               hidden_channels=args.hyper_hidden,
                                               out_channels=1,
                                               num_layers=args.hyper_num_layers,
                                               dropout=0.5, args=args).to(self.device)
                self.gate_model = [self.para1_model, self.parabias1_model, self.para2_model, self.parabias2_model]
            elif args.original_data == "hypergcn":
                pass
        elif args.biased == "none":
            self.gate_model = MLP(input_dim, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)

        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)
        if self.args.subloss == 'loss_combine':
            if self.args.dataset in ['genius', 'ogbn-genius']:
                self.crit = nn.BCEWithLogitsLoss(reduction='none')
                self.crit_pretrain = nn.BCEWithLogitsLoss()
            else:
                self.args.crit = 'nllloss'
                self.crit = nn.NLLLoss(reduction='none')
                self.crit_pretrain = nn.NLLLoss()
        elif self.args.subloss == 'combine_loss':
            if self.args.dataset in ['genius', 'ogbn-genius']:
                self.crit = nn.BCEWithLogitsLoss()
                self.crit_pretrain = nn.BCEWithLogitsLoss()
            else:
                self.args.crit = 'crossentropy'
                self.crit = nn.CrossEntropyLoss()
                self.crit_pretrain = nn.CrossEntropyLoss()

        if self.args.adam:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            if self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.Adam(
                self.model1.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.Adam(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            elif self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.AdamW(
                self.model1.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.AdamW(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model1.reset_parameters()
        self.model2.reset_parameters()
        if self.args.original_data != "hypermlp":
            self.gate_model.reset_parameters()
        else:
            self.para1_model.reset_parameters()
            self.parabias1_model.reset_parameters()
            self.para2_model.reset_parameters()
            self.parabias2_model.reset_parameters()

        if self.args.submethod in ['pretrain_model1', 'pretrain_both']:
            if self.args.model1 != self.args.model2:
                _ = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                               self.optimizer1, self.split_idx, self.evaluator)
            else:
                _ = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                               self.optimizer1, self.split_idx, self.evaluator, 1)
        if self.args.submethod in ['pretrain_model2', 'pretrain_both']:
            if self.args.model1 != self.args.model2:
                _ = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                               self.optimizer2, self.split_idx, self.evaluator)
            else:
                _ = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                               self.optimizer2, self.split_idx, self.evaluator, 2)

        self.model1.dropout = self.args.dropout_gate
        self.model2.dropout = self.args.dropout_gate

        if self.args.original_data != "hypermlp":
            self.gate_model.dropout = self.args.dropout_gate

        result = simple_gate_train_test_wrapper(self.args, self.model1, self.model2, self.gate_model,
                                                self.data, self.crit, self.optimizer, self.split_idx,
                                                self.evaluator, self.device)

        return result


class SimpleGateInTurn(object):
    def __init__(self, args, device, data, dataset, split_idx):

        self.args = args
        self.device = device
        self.data = data.to(self.device)
        model1_hyper, model2_hyper = get_model_hyperparameters(args, data, dataset)
        input_dim, hidden_dim1, output_dim, num_layers1 = model1_hyper
        input_dim, hidden_dim2, output_dim, num_layers2 = model2_hyper
        dropout = args.dropout

        self.model1 = load_model(args.model1, input_dim, hidden_dim1, output_dim, num_layers1, dropout, args).to(
            self.device)
        self.model2 = load_model(args.model2, input_dim, hidden_dim2, output_dim, num_layers2, dropout, args).to(
            self.device)

        if args.biased == "logit":
            if args.original_data == "true":
                self.gate_model = GateMLP(input_dim + output_dim, hidden_dim1, 1, num_layers1, dropout, args).to(
                    self.device)
            elif args.original_data == "false":
                self.gate_model = GateMLP(output_dim, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        elif args.biased == "dispersion":
            if args.original_data == "true":
                self.gate_model = GateMLP(input_dim + 2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
            elif args.original_data == "false":
                self.gate_model = GateMLP(2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
            elif args.original_data == "hypermlp":
                self.para1_model = GateMLP(in_channels=input_dim,
                                           hidden_channels=args.hyper_hidden,
                                           out_channels=2 * args.model1_hidden_dim,
                                           num_layers=args.hyper_num_layers,
                                           dropout=0.5, args=args).to(self.device)
                self.parabias1_model = GateMLP(in_channels=input_dim,
                                               hidden_channels=args.hyper_hidden,
                                               out_channels=args.model1_hidden_dim,
                                               num_layers=args.hyper_num_layers,
                                               dropout=0.5, args=args).to(self.device)
                self.para2_model = GateMLP(in_channels=input_dim,
                                           hidden_channels=args.hyper_hidden,
                                           out_channels=args.model1_hidden_dim,
                                           num_layers=args.hyper_num_layers,
                                           dropout=0.5, args=args).to(self.device)
                self.parabias2_model = GateMLP(in_channels=input_dim,
                                               hidden_channels=args.hyper_hidden,
                                               out_channels=1,
                                               num_layers=args.hyper_num_layers,
                                               dropout=0.5, args=args).to(self.device)
                self.gate_model = [self.para1_model, self.parabias1_model, self.para2_model, self.parabias2_model]
        elif args.biased == "none":
            self.gate_model = MLP(input_dim, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)

        if self.args.dataset in ['genius', 'ogbn-genius']:
            self.crit = nn.BCEWithLogitsLoss(reduction='none')
            self.crit_pretrain = nn.BCEWithLogitsLoss()
        else:
            if self.args.crit == 'nllloss':
                self.crit = nn.NLLLoss(reduction='none')
                self.crit_pretrain = nn.NLLLoss()
            else:
                raise ValueError('Invalid Crit. In In Turn setting, only NLLLoss can be used')

        if self.args.adam:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            if self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.Adam(
                list(self.model1.parameters()) + list(self.gate_model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.Adam(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            elif self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.AdamW(
                list(self.model1.parameters()) + list(self.gate_model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.AdamW(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model1.reset_parameters()
        self.model2.reset_parameters()
        if self.args.original_data != "hypermlp":
            self.gate_model.reset_parameters()
        else:
            self.para1_model.reset_parameters()
            self.parabias1_model.reset_parameters()
            self.para2_model.reset_parameters()
            self.parabias2_model.reset_parameters()

        if self.args.submethod in ['pretrain_model1', 'pretrain_both']:
            _ = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                           self.optimizer1, self.split_idx, self.evaluator)
        if self.args.submethod in ['pretrain_model2', 'pretrain_both']:
            _ = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                           self.optimizer2, self.split_idx, self.evaluator)
            print('Model pretraining done')
        self.model1.dropout = self.args.dropout_gate
        self.model2.dropout = self.args.dropout_gate
        if self.args.original_data != "hypermlp":
            self.gate_model.dropout = self.args.dropout_gate
        result = mowse_train_test_wrapper_simple_gate(self.args, self.model1, self.model2, self.gate_model, self.data,
                                                      self.crit,
                                                      self.optimizer1, self.optimizer2,
                                                      self.split_idx, self.evaluator, self.device)

        return result


def train_simple_gate(model1, model2, gate_model, data, crit, optimizer, args):
    model1.train()
    model2.train()
    if args.original_data != "hypermlp":
        gate_model.train()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.train()
        parabias1_model.train()
        para2_model.train()
        parabias2_model.train()

    optimizer.zero_grad()

    model1_out = model1(data)
    model2_out = model2(data)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)


    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))

    if args.subloss == 'combine_loss':
        # out = F.softmax(model1_out, dim = 1) * gating + F.softmax(model2_out, dim = 1) * (1 - gating)
        out = model1_out * gating + model2_out * (1 - gating)
        loss = crit(out[data.train_mask], data.y.squeeze(1)[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    elif args.subloss == 'loss_combine':
        if crit.__class__.__name__ == 'NLLLoss':
            loss1 = crit(F.log_softmax(model1_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
            loss2 = crit(F.log_softmax(model2_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
        elif crit.__class__.__name__ == 'CrossEntropyLoss':
            loss1 = crit(model1_out[data.train_mask], data.y.squeeze(1)[data.train_mask])
            loss2 = crit(model2_out[data.train_mask], data.y.squeeze(1)[data.train_mask])
        gating = gating[data.train_mask].view(-1)
        loss = loss1 * gating + loss2 * (1 - gating)
        loss.mean().backward()
        optimizer.step()
        return loss.mean().item()


def eval_for_simplicity(evaluator, data, model1_out, model2_out, gating, split_idx, args):
    out = model1_out * gating + model2_out * (1 - gating)

    y_pred = out.argmax(dim=-1, keepdim=True)
    model1_weight = gating.mean().item()

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_acc, valid_acc, test_acc, model1_weight


def eval_for_multi(args, gating, device, model1_out, model2_out, evaluator, data, split_idx):
    tmp_train_acc_list = []
    tmp_val_acc_list = []
    tmp_test_acc_list = []
    model1_weight_list = []
    for t in range(args.m_times):
        m = torch.rand(gating.shape).to(device)
        gate = (m < gating).int().view(-1, 1)

        model1_pred = model1_out.argmax(dim=-1, keepdim=True)
        model2_pred = model2_out.argmax(dim=-1, keepdim=True)
        y_pred = model1_pred.view(-1) * gate.view(-1) + model2_pred.view(-1) * (1 - gate.view(-1))
        y_pred = y_pred.view(-1, 1)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })

        tmp_train_acc_list.append(train_acc)
        tmp_val_acc_list.append(valid_acc)
        tmp_test_acc_list.append(test_acc)
        gate = gate.view(-1)
        model1_weight = gate.sum().item() / len(gate)
        model1_weight_list.append(model1_weight)

    train_acc = np.mean(tmp_train_acc_list)
    valid_acc = np.mean(tmp_val_acc_list)
    test_acc = np.mean(tmp_test_acc_list)
    model1_weight = np.mean(model1_weight_list)
    return train_acc, valid_acc, test_acc, model1_weight


@torch.no_grad()
def test_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = F.softmax(model1(data), dim=1)
    model2_out = F.softmax(model2(data), dim=1)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))

    if args.infer_method == 'simple':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_simplicity(evaluator, data, model1_out, model2_out,
                                                                            gating, split_idx, args)
        return train_acc, valid_acc, test_acc, model1_weight
    elif args.infer_method == 'multi':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_multi(args, gating, device, model1_out, model2_out,
                                                                       evaluator, data, split_idx)
        return train_acc, valid_acc, test_acc, model1_weight


def simple_gate_train_test_wrapper(args, model1, model2, gate_model, data, crit, optimizer, split_idx, evaluator,
                                   device):
    check = 0
    best_score = 0
    best_loss = float('-inf')

    for i in range(args.epoch):
        loss = train_simple_gate(model1, model2, gate_model, data, crit, optimizer, args)
        result = test_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        val_loss, gate_weight, l1, l2, g = cal_val_loss_simple_gate(model1, model2, gate_model, data, crit, args)
        val_loss = - val_loss

        model1_result = test_vanilla(model1, data, split_idx, evaluator, args)
        model2_result = test_vanilla(model2, data, split_idx, evaluator, args)

        if args.log:
            wandb.log({'Train Acc': result[0],
                       'Val Acc': result[1],
                       'Test Acc': result[2],
                       'Train Loss': loss,
                       'Val Loss': -val_loss,
                       'Gate Weight': gate_weight,
                       "Gate Weight Dis": g,
                       'Model1 Val Acc': model1_result[1],
                       'Model2 Val Acc': model2_result[2],
                       'L1': l1,
                       'L2': l2})

        if i % args.print_freq == 0:
            print('{} epochs trained, loss {:.4f}'.format(i, loss))

        if args.no_early_stop is False:
            # if result[1] > best_score:
            if val_loss > best_loss:
                check = 0
                best_score = result[1]
                best_loss = val_loss
                saved_model1 = save_model(args, model1, 'model1')
                saved_model2 = save_model(args, model2, 'model2')
                if args.original_data != "hypermlp":
                    saved_gate_model = save_model(args, gate_model, 'gate_model')
                else:
                    para1_model, parabias1_model, para2_model, parabias2_model = gate_model
                    saved_para1_model = save_model(args, para1_model, 'para1_model')
                    save_parabias1_model = save_model(args, parabias1_model, 'parabias1_model')
                    saved_para2_model = save_model(args, para2_model, 'para2_model')
                    save_parabias2_model = save_model(args, parabias2_model, 'parabias2_model')
            else:
                check += 1
                if check > args.patience:
                    print("{} epochs trained, best val loss {:.4f}".format(i, -best_loss))
                    break
        else:
            # if result[1] > best_score:
            if val_loss > best_loss:
                check = 0
                best_score = result[1]
                best_loss = val_loss
                saved_model1 = save_model(args, model1, 'model1')
                saved_model2 = save_model(args, model2, 'model2')
                if args.original_data != "hypermlp":
                    saved_gate_model = save_model(args, gate_model, 'gate_model')
                else:
                    para1_model, parabias1_model, para2_model, parabias2_model = gate_model
                    saved_para1_model = save_model(args, para1_model, 'para1_model')
                    save_parabias1_model = save_model(args, parabias1_model, 'parabias1_model')
                    saved_para2_model = save_model(args, para2_model, 'para2_model')
                    save_parabias2_model = save_model(args, parabias2_model, 'parabias2_model')

    model1.load_state_dict(torch.load(saved_model1))
    model2.load_state_dict(torch.load(saved_model2))
    if args.original_data != "hypermlp":
        gate_model.load_state_dict(torch.load(saved_gate_model))
    else:
        para1_model.load_state_dict(torch.load(saved_para1_model))
        parabias1_model.load_state_dict(torch.load(save_parabias1_model))
        para2_model.load_state_dict(torch.load(saved_para2_model))
        parabias2_model.load_state_dict(torch.load(save_parabias2_model))
        gate_model = [para1_model, parabias1_model, para2_model, parabias2_model]

    result = test_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
    train_acc, val_acc, test_acc, model1_weight = result
    print('Final results: Train {:.2f} Val {:.2f} Test {:.2f} Model1 Weight {:.2f}'.format(train_acc * 100,
                                                                                           val_acc * 100,
                                                                                           test_acc * 100,
                                                                                           model1_weight * 100))

    return train_acc, val_acc, test_acc, model1_weight, best_loss


@torch.no_grad()
def cal_val_loss_simple_gate(model1, model2, gate_model, data, crit, args):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()

    model1_out = model1(data)
    model2_out = model2(data)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)

    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))

    loss1 = crit(F.log_softmax(model1_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
    loss2 = crit(F.log_softmax(model2_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
    if args.subloss == 'combine_loss':
        out = model1_out * gating + model2_out * (1 - gating)
        loss = crit(F.log_softmax(out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])

    elif args.subloss == 'loss_combine':
        loss1 = loss1.mean()
        loss2 = loss2.mean()
        loss = (loss1 * gating[data.val_mask] + loss2 * (1 - gating[data.val_mask])).mean()

    return loss.item(), gating.mean().item(), loss1.item(), loss2.item(), gating.cpu().numpy()


def mowse_train_test_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer1, optimizer2,
                                         split_idx, evaluator, device):
    big_epoch_check = 0
    big_best_score = 0
    # big_best_loss = float('-inf')
    n = len(data.y)
    df = pd.DataFrame({"id":list(range(n))})


    for j in range(args.big_epoch):
        print('------ Big epoch {} Model 1 ------'.format(j))
        model1_turn_val_acc = mowse_train_test_model1_turn_wrapper_simple_gate(args, model1, model2, gate_model, data,
                                                                               crit, optimizer1,
                                                                               split_idx, evaluator, device,
                                                                               big_best_score)
        # locals()[f"mlp_{j}_l1"] = loss1
        # locals()[f"mlp_{j}_l2"] = loss2
        # locals()[f"mlp_{j}_g"] = gating
        # locals()[f"mlp_{j}_c"] = correct
        # df.loc[:, f"mlp_{j}_l1"] = loss1
        # df.loc[:, f"mlp_{j}_l2"] = loss2
        # df.loc[:, f"mlp_{j}_g"] = gating
        # df.loc[:, f"mlp_{j}_c"] = correct


        print('------ Big epoch {} Model 2 ------'.format(j))
        mowse_train_test_model2_turn_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer2,
                                                         split_idx, evaluator, device,
                                                         model1_turn_val_acc)

        if j % 1 == 0:
            model2_emb, gating = generate_embedding(args, model1, model2, gate_model, data)


            ncol = model2_emb.shape[1]
            df = pd.DataFrame(data=model2_emb, columns=[f"col{_}" for _ in range(ncol)])
            df.loc[:, "confidence"] = gating
            df.to_pickle(args.denoise_save_path + f"dataemb_{args.dataset}_{j}.pkl")

        # locals()[f"gnn_{j}_l1"] = loss1
        # locals()[f"gnn_{j}_l2"] = loss2
        # locals()[f"gnn_{j}_g"] = gating
        # locals()[f"gnn_{j}_c"] = correct

        # df.loc[:, f"gnn_{j}_l1"] = loss1
        # df.loc[:, f"gnn_{j}_l2"] = loss2
        # df.loc[:, f"gnn_{j}_g"] = gating
        # df.loc[:, f"gnn_{j}_c"] = correct

        # df.to_pickle(args.denoise_save_path + f"denoise_{args.dataset}.pkl")


        result = test_mowse_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        # val_loss, VAL_LOSS, LOSS1, LOSS2 = loss_mowse_simple_gate(model1, model2, gate_model, data, crit, args)
        # val_loss = - val_loss
        log_conf = get_cur_confidence_simple_gate(args, gate_model, model1, data)
        if args.log:
            wandb.log({'Train Acc': result[0],
                       'Val Acc': result[1],
                       'Test Acc': result[2],
                       'Confidence': log_conf.mean().item(),
                       "Confidence Distribution": log_conf.cpu().numpy()
                       })

        if args.no_early_stop is False:
            # if val_loss > big_best_loss:
            if result[1] > big_best_score:
                big_epoch_check = 0
                big_best_score = result[1]
                # big_best_loss = val_loss
                saved_model1_big = save_moe_model(args, model1, '1_big')
                saved_model2_big = save_moe_model(args, model2, '2_big')
            else:
                big_epoch_check += 1
                if big_epoch_check > args.big_patience:
                    print("{} big epochs trained, best val acc {:.4f}".format(j, big_best_score))
                    break
        else:
            # if val_loss > big_best_loss:
            if result[1] > big_best_score:
                big_epoch_check = 0
                big_best_score = result[1]
                # big_best_loss = val_loss
                saved_model1_big = save_moe_model(args, model1, '1_big')
                saved_model2_big = save_moe_model(args, model2, '2_big')

    model1.load_state_dict(torch.load(saved_model1_big))
    model2.load_state_dict(torch.load(saved_model2_big))
    result = test_mowse_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
    train_acc, val_acc, test_acc, model1_weight, _ = result
    print('Final results: Train {:.2f} Val {:.2f} Test {:.2f} Model1 Weight {:.2f}'.format(train_acc * 100,
                                                                                           val_acc * 100,
                                                                                           test_acc * 100,
                                                                                           model1_weight * 100))
    return result


@torch.no_grad()
def generate_embedding(args, model1, model2, gate_model, data):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = model1(data)
    model2_out, model2_emb = model2(data, mode=True)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))

    return model2_emb.cpu().numpy(), gating.view(-1).cpu().numpy()






def mowse_train_test_model1_turn_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer1,
                                                     split_idx, evaluator, device,
                                                     big_best_score):
    saved_model1_previous_big_turn = save_moe_model(args, model1, '1_big')
    saved_model2_previous_big_turn = save_moe_model(args, model2, '2_big')

    check = 0
    best_score = 0
    # best_loss = float('-inf')
    for i in range(args.epoch):

        loss, l1, l2 = train_mowse1_simple_gate(model1, model2, gate_model, data, crit, optimizer1, args)
        result = test_mowse_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        # val_loss = loss_mowse_simple_gate(model1, model2, gate_model, data, crit, args)
        # val_loss = - val_loss
        if args.log:
            wandb.log({'L1 train loss': l1.mean().item(),
                       "L1 train loss Distribution": l1.detach().cpu().numpy(),
                       'L2 train loss': l2.mean().item(),
                       "L2 train loss Distribution": l2.detach().cpu().numpy()
                       })

        if i % args.print_freq == 0:
            print('{} epochs trained, loss {:.4f}'.format(i, loss))

        # if val_loss > best_loss:
        if result[1] > best_score:
            check = 0
            # best_loss = val_loss
            best_score = result[1]
            saved_model1 = save_moe_model(args, model1, '1_inner')
            saved_model2 = save_moe_model(args, model2, '2_inner')
        else:
            check += 1
            if check > args.patience:
                print("{} epochs trained, best val loss {:.4f}".format(i, best_score))
                break

    if best_score > big_best_score:
        model1.load_state_dict(torch.load(saved_model1))
        model2.load_state_dict(torch.load(saved_model2))
    else:
        model1.load_state_dict(torch.load(saved_model1_previous_big_turn))
        model2.load_state_dict(torch.load(saved_model2_previous_big_turn))

    # loss1, loss2, gating, correct = get_denoise_info(model1, model2, gate_model, data, crit, args, evaluator, split_idx, device)

    saved_model1 = save_moe_model(args, model1, 1)
    saved_model2 = save_moe_model(args, model2, 2)
    return max(best_score, big_best_score)


def mowse_train_test_model2_turn_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer2,
                                                     split_idx, evaluator, device,
                                                     model1_turn_val_acc):
    saved_model1_previous_small_turn = save_moe_model(args, model1, 1)
    saved_model2_previous_small_turn = save_moe_model(args, model2, 2)

    check = 0
    best_score = 0
    # best_loss = float('-inf')
    for i in range(args.epoch):

        loss, l1, l2 = train_mowse2_simple_gate(model1, model2, gate_model, data, crit, optimizer2, args)
        result = test_mowse_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        # val_loss = loss_mowse_simple_gate(model1, model2, gate_model, data, crit, args)
        # val_loss = - val_loss

        if args.log:
            wandb.log({'L1 train loss': l1.mean().item(),
                       "L1 train loss Distribution": l1.detach().cpu().numpy(),
                       'L2 train loss': l2.mean().item(),
                       "L2 train loss Distribution": l2.detach().cpu().numpy()
                       })

        if i % args.print_freq == 0:
            print('{} epochs trained, loss {:.4f}'.format(i, loss))

        # if val_loss > best_loss:
        if result[1] > best_score:
            check = 0
            best_score = result[1]
            # best_loss = val_loss
            saved_model1 = save_moe_model(args, model1, '1_inner')
            saved_model2 = save_moe_model(args, model2, '2_inner')
        else:
            check += 1
            if check > args.patience:
                print("{} epochs trained, best val acc {:.4f}".format(i, best_score))
                break
    if best_score > model1_turn_val_acc:
        model1.load_state_dict(torch.load(saved_model1))
        model2.load_state_dict(torch.load(saved_model2))
    else:
        model1.load_state_dict(torch.load(saved_model1_previous_small_turn))
        model2.load_state_dict(torch.load(saved_model2_previous_small_turn))
    # loss1, loss2, gating, correct = get_denoise_info(model1, model2, gate_model, data, crit, args, evaluator, split_idx,
    #                                                  device)
    saved_model1 = save_moe_model(args, model1, 1)
    saved_model2 = save_moe_model(args, model2, 2)
    # return loss1, loss2, gating, correct

@torch.no_grad()
def get_denoise_info(model1, model2, gate_model, data, crit, args, evaluator, split_idx, device):
    model1.eval()
    model2.eval()
    # get loss info

    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()

    model1_out = model1(data)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))
    with torch.no_grad():
        model2_out = model2(data)

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1), data.y.squeeze(1))
        loss2 = crit(F.log_softmax(model2_out, dim=1), data.y.squeeze(1))
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating.view(-1)
    loss = loss1 * gating + loss2 * (1 - gating)



    model1_out = F.softmax(model1_out, dim=1)
    model2_out = F.softmax(model2_out, dim=1)

    if args.infer_method == 'simple':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_simplicity(evaluator, data, model1_out, model2_out,
                                                                            gating, split_idx, args)
        result = (train_acc, valid_acc, test_acc, model1_weight, float(0))
    elif args.infer_method == 'multi':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_multi(args, gating, device, model1_out, model2_out,
                                                                       evaluator, data, split_idx)
        result = (train_acc, valid_acc, test_acc, model1_weight, float(0))



    m = torch.rand(gating.shape).to(device)
    gate = (m < gating).int().view(-1, 1)

    model1_pred = model1_out.argmax(dim=-1, keepdim=True)
    model2_pred = model2_out.argmax(dim=-1, keepdim=True)
    y_pred = model1_pred.view(-1) * gate.view(-1) + model2_pred.view(-1) * (1 - gate.view(-1))
    y_pred = y_pred.view(-1, 1)
    correct = (y_pred == data.y).view(-1).float()



    return loss1.cpu().numpy(), loss2.cpu().numpy(), gating.cpu().numpy(), correct.cpu().numpy()







def train_mowse1_simple_gate(model1, model2, gate_model, data, crit, optimizer1, args):
    model1.train()
    model2.train()
    if args.original_data != "hypermlp":
        gate_model.train()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.train()
        parabias1_model.train()
        para2_model.train()
        parabias2_model.train()
    optimizer1.zero_grad()
    model1_out = model1(data)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))
    with torch.no_grad():
        model2_out = model2(data)

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
        loss2 = crit(F.log_softmax(model2_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating[data.train_mask].view(-1)
    loss = loss1 * gating + loss2 * (1 - gating)
    loss.mean().backward()
    optimizer1.step()
    return loss.mean().item(), loss1, loss2


def train_mowse2_simple_gate(model1, model2, gate_model, data, crit, optimizer2, args):
    model1.train()
    model2.train()
    if args.original_data != "hypermlp":
        gate_model.train()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.train()
        parabias1_model.train()
        para2_model.train()
        parabias2_model.train()
    optimizer2.zero_grad()
    model2_out = model2(data)

    with torch.no_grad():
        model1_out = model1(data)
        if args.biased == "logit":
            if args.original_data == "true":
                x = torch.cat((model1_out, data.mlp_x), dim=1)
                gating = nn.Sigmoid()(gate_model(x))
            elif args.original_data == "false":
                gating = nn.Sigmoid()(gate_model(model1_out))
        elif args.biased == 'dispersion':
            var_conf = compute_confidence(model1_out, "variance")
            ent_conf = compute_confidence(model1_out, "entropy")
            dispersion = torch.cat((var_conf, ent_conf), dim=1)
            if args.original_data == "true":
                gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
                gating = nn.Sigmoid()(gate_model(gate_input))
            elif args.original_data == "false":
                gating = nn.Sigmoid()(gate_model(dispersion))
            elif args.original_data == "hypermlp":
                node_feature = data.mlp_x
                para1 = para1_model(node_feature)
                parabias1 = parabias1_model(node_feature)
                para2 = para2_model(node_feature)
                parabias2 = parabias2_model(node_feature)

                para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
                dispersion = dispersion[:, np.newaxis, :]
                para2 = para2[:, :, np.newaxis]

                gating = torch.matmul(dispersion, para1)
                gating += parabias1[:, np.newaxis, :]
                gating = torch.matmul(gating, para2)
                gating += parabias2[:, np.newaxis, :]
                gating = nn.Sigmoid()(gating).view(-1, 1)
        elif args.biased == "none":
            gating = nn.Sigmoid()(gate_model(data))

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
        loss2 = crit(F.log_softmax(model2_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating[data.train_mask].view(-1)

    loss = loss1 * gating + loss2 * (1 - gating)
    loss.mean().backward()
    optimizer2.step()
    return loss.mean().item(), loss1, loss2


@torch.no_grad()
def loss_mowse_simple_gate(model1, model2, gate_model, data, crit, args):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = model1(data)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))
    with torch.no_grad():
        model2_out = model2(data)

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
        loss2 = crit(F.log_softmax(model2_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating[data.val_mask].view(-1)
    loss = loss1 * gating + loss2 * (1 - gating)

    return loss.mean().item(), loss, loss1, loss2


@torch.no_grad()
def test_mowse_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = model1(data)
    model2_out = model2(data)
    if args.biased == 'logit':
        if args.original_data == "true":
            x = torch.cat((model1_out, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(x))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(model1_out))
    elif args.biased == 'dispersion':
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    elif args.biased == 'none':
        gating = nn.Sigmoid()(gate_model(data))

    model1_out = F.softmax(model1_out, dim=1)
    model2_out = F.softmax(model2_out, dim=1)

    if args.infer_method == 'simple':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_simplicity(evaluator, data, model1_out, model2_out,
                                                                            gating, split_idx, args)
        return train_acc, valid_acc, test_acc, model1_weight, float(0)
    elif args.infer_method == 'multi':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_multi(args, gating, device, model1_out, model2_out,
                                                                       evaluator, data, split_idx)
        return train_acc, valid_acc, test_acc, model1_weight, float(0)

    # train_acc, valid_acc, test_acc, model1_weight = eval_for_multi(args, gating, device, model1_out, model2_out,
    #                                                                evaluator, data, split_idx)
    # return train_acc, valid_acc, test_acc, model1_weight, float(0)


@torch.no_grad()
def get_cur_confidence_simple_gate(args, gate_model, model1, data):
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1.eval()
    with torch.no_grad():
        if args.biased == 'logit':
            model1_out = model1(data)
            if args.original_data == "true":
                x = torch.cat((model1_out, data.mlp_x), dim=1)
                gating = nn.Sigmoid()(gate_model(x))
            elif args.original_data == "false":
                gating = nn.Sigmoid()(gate_model(model1_out))
        elif args.biased == 'dispersion':
            model1_out = model1(data)
            var_conf = compute_confidence(model1_out, "variance")
            ent_conf = compute_confidence(model1_out, "entropy")
            dispersion = torch.cat((var_conf, ent_conf), dim=1)
            if args.original_data == "true":
                gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
                gating = nn.Sigmoid()(gate_model(gate_input))
            elif args.original_data == "false":
                gating = nn.Sigmoid()(gate_model(dispersion))
            elif args.original_data == "hypermlp":
                node_feature = data.mlp_x
                para1 = para1_model(node_feature)
                parabias1 = parabias1_model(node_feature)
                para2 = para2_model(node_feature)
                parabias2 = parabias2_model(node_feature)

                para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
                dispersion = dispersion[:, np.newaxis, :]
                para2 = para2[:, :, np.newaxis]

                gating = torch.matmul(dispersion, para1)
                gating += parabias1[:, np.newaxis, :]
                gating = torch.matmul(gating, para2)
                gating += parabias2[:, np.newaxis, :]
                gating = nn.Sigmoid()(gating).view(-1, 1)
        elif args.biased == 'none':
            gating = nn.Sigmoid()(gate_model(data))
    return gating.view(-1)