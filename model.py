import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, MessagePassing
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(GCN, self).__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=~args.no_cached, add_self_loops=~args.no_add_self_loops,
                    normalize=~args.no_normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=~args.no_cached,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=~args.no_cached, add_self_loops=~args.no_add_self_loops,
                    normalize=~args.no_normalize))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm is False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final


class Sage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm is False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final





class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(MLP(in_channels, hidden_channels, hidden_channels, num_layers, dropout, args, gin=True)))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout, args, gin=True)))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GINConv(MLP(hidden_channels, out_channels, hidden_channels, num_layers, dropout, args, gin=True)))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        print(out_channels)
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=args.heads, cached=~args.no_cached,
                                  add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * args.heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=args.heads, cached=~args.no_cached,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * args.heads))
        if args.dataset == "pubmed":
            self.convs.append(
                GATConv(hidden_channels * args.heads, out_channels, heads=1, cached=~args.no_cached, concat=False,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
        else:
            self.convs.append(
                GATConv(hidden_channels * args.heads, out_channels, heads=1, cached=~args.no_cached, concat=False,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args, gin=False):
        super(MLP, self).__init__()
        self.args = args
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.gin = gin
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        if self.gin:
            x = data
        else:
            x = data.mlp_x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lins[-1](x)
        return x_final


class GateMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        self.args = args
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.args.no_batch_norm is False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lins[-1](x)
        return x_final


class MLPLearn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(MLPLearn, self).__init__()
        self.args = args

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.conf_layer1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conf_layer2 = torch.nn.Linear(hidden_channels, 1)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.mlp_x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lins[-1](x)
        conf = self.conf_layer1(x)
        conf = F.relu(conf)
        conf = F.dropout(conf, p=self.dropout, training=self.training)
        conf = torch.nn.Sigmoid()(self.conf_layer2(conf))
        return x_final, conf


class GCN_SpMoE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_experts=4, k=1, coef=1e-2):
        super(GCN_SpMoE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        # self.convs.append(
            # GCNConv(in_channels, hidden_channels, normalize=True))
        for layer_idx in range(1):
            if layer_idx % 2 == 0:
                ffn = MoE(input_size=in_channels, output_size=hidden_channels, num_experts=num_experts, k=k,
                          coef=coef)
                self.convs.append(ffn)
            else:
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, normalize=True))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=True))

        # print(self.convs)

        self.dropout = dropout

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        self.load_balance_loss = 0  # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for conv in self.convs[:-1]:
            if isinstance(conv, MoE):
                x, _layer_load_balance_loss = conv(x, adj_t)
                self.load_balance_loss += _layer_load_balance_loss
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # self.load_balance_loss /= math.ceil((self.num_layers - 2) / 2)
        return x

class SAGE_SpMoE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_experts=4, k=1, coef=1e-2):
        super(SAGE_SpMoE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        # self.convs.append(SAGEConv(in_channels, hidden_channels))
        for layer_idx in range(1):
            if layer_idx % 2 == 0:
                ffn = MoE(input_size=in_channels, output_size=hidden_channels, num_experts=num_experts, k=k,
                          coef=coef, sage=True)
                self.convs.append(ffn)
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout


    def reset_parameters(self):
        pass
        # for conv in self.convs:
        #     conv.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        self.load_balance_loss = 0  # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for conv in self.convs[:-1]:
            if isinstance(conv, MoE):
                x, _layer_load_balance_loss = conv(x, adj_t)
                self.load_balance_loss += _layer_load_balance_loss
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # self.load_balance_loss /= math.ceil((self.num_layers - 2) / 2)
        return x

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp, edge_index, edge_attr):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # Note by Haotao:
        # self._batch_index: shape=(N_batch). The re-order indices from 0 to N_batch-1.
        # inp_exp: shape=inp.shape. The input Tensor re-ordered by self._batch_index along the batch dimension.
        # self._part_sizes: shape=(N_experts), sum=N_batch. self._part_sizes[i] is the number of samples routed towards expert[i].
        # return value: list [Tensor with shape[0]=self._part_sizes[i] for i in range(N_experts)]

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        edge_index_exp = edge_index[:,self._batch_index]
        edge_attr_exp = edge_attr[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(edge_index_exp, self._part_sizes, dim=1), torch.split(edge_attr_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, noisy_gating=True, k=4, coef=1e-2, sage=False):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        # instantiate experts
        conv = SAGEConv if sage else GCNConv
        self.experts = nn.ModuleList([conv(input_size, output_size, normalize=True) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, edge_index, edge_attr=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef



        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](x, edge_index, edge_attr)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]

        # gates: shape=[num_nodes, num_experts]
        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.mean(dim=1)

        return y, loss


class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init="PPR", Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):

        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)



        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR', dprate=.0, dropout=.5, K=10, alpha=.1,
                 Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()

        self.mlp = GPR_MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.edge_index
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GPR_MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)



class AdaGCN(nn.Module):
    def __init__(self, nfeat,  nhid, nclass, dropout, dropout_adj):
        super().__init__()
        fcs = [MixedLinear(nfeat, nhid, bias=False)]
        fcs.append(nn.Linear(nhid, nclass, bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())

        if dropout == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout) # p: drop rate
        if dropout_adj == 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj) # p: drop rate
        self.act_fn = nn.ReLU()

    def reset_parameters(self):
        for lin in self.fcs:
            lin.reset_parameters()

    def _transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, x, adj):  # X, A
        logits = self._transform_features(x) # MLP: X->H, Mixed-layer + some layers FC
        return F.log_softmax(logits, dim=-1)


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class SparseMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.save_for_backward(sparse, dense)
        return torch.mm(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        sparse, dense = ctx.saved_tensors
        grad_sparse = grad_dense = None
        if ctx.needs_input_grad[0]:
            grad_sparse = torch.mm(grad_output, dense.t())
        if ctx.needs_input_grad[1]:
            grad_dense = torch.mm(sparse.t(), grad_output)
        return grad_sparse, grad_dense
