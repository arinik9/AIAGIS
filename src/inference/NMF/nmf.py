import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


dt = 1

# # define the model architecture
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size)
#         self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input, hidden):
#         output, hidden = self.lstm(input, hidden)
#         output = self.linear(output[-1])
#         return output, hidden


# LSTM from scratch: https://rodriguesthiago.me/posts/recurrent_neural_networks_rnn_lstm_and_gru/#lstm-long-short-term-memory
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for the input gate
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # Parameters for the forget gate
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # Parameters for the input node
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # Parameters for the output gate
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self):
        # Initializes parameters with suitable distributions
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, xinput, pre_c, pre_h):
        """
        Forward method to process a sequence of inputs over time.
        :param xinput: input tensor.
        :param init_states: Initial states (hidden state and cell state).
        :return: Outputs over time and the last state (hidden state and cell state).
        """
        c_t = pre_c
        h_t = pre_h

        # Input gate: decides which information will be updated
        i_t = torch.sigmoid(xinput @ self.W_ii.t() +
                            self.b_ii + h_t @ self.W_hi.t() + self.b_hi)

        # Forget gate: decides which information will be discarded from the cell state
        f_t = torch.sigmoid(xinput @ self.W_if.t() +
                            self.b_if + h_t @ self.W_hf.t() + self.b_hf)

        # Input node: creates a vector of new candidates to be added to the cell state
        g_t = torch.tanh(xinput @ self.W_ig.t() + self.b_ig +
                         h_t @ self.W_hg.t() + self.b_hg)

        # Updating the internal state of the memory cell
        c_t = f_t * c_t + i_t * g_t

        # Output gate: decides which parts of the cell state will be outputs
        o_t = torch.sigmoid(xinput @ self.W_io.t() +
                            self.b_io + h_t @ self.W_ho.t() + self.b_ho)

        # Hidden state: is the output of the LSTM, using the cell state passed through a tanh function
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t



def get_weights_with_regularization(shape, regularizer):
    device = "mps"
    initializer=torch.normal(mean=0.0, std=0.00001, size=shape)
    weights = torch.tensor(initializer, requires_grad=True, device=device, dtype=torch.float)
    if regularizer != None:
        return regularizer(weights)
    return weights


def inf_erro_CNN(input_tensor, nNodes, regularizer):
    #the inner neural network to learn h
    #note that： in each timestep, we share the same parameters.
    INPUT_NODE=nNodes
    OUTPUT_NODE = nNodes
    LAYER1_NODE= nNodes
    LAYER2_NODE=nNodes
    # LAYER 1
    weights = nn.Parameter(get_weights_with_regularization([INPUT_NODE, LAYER1_NODE], regularizer))
    biases = nn.Parameter(torch.zeros([LAYER1_NODE]))
    layer1 = F.elu(torch.matmul(input_tensor, weights) + biases)
    # LAYER 2
    weights = nn.Parameter(get_weights_with_regularization([LAYER1_NODE, LAYER2_NODE], regularizer))
    biases = nn.Parameter(torch.zeros([LAYER2_NODE]))
    layer2 = F.elu(torch.matmul(layer1, weights) + biases)
    # LAYER 3
    weights = nn.Parameter(get_weights_with_regularization([LAYER2_NODE, OUTPUT_NODE], regularizer))
    biases = nn.Parameter(torch.zeros([LAYER2_NODE]))
    layer3 = F.elu(torch.matmul(layer2, weights) + biases)
    return layer3

def inf_LSTM_MTM(xinput,nNodes,pre_c,pre_h,regularizer,forget_bias=1.0):
    # Input size and hidden size for demonstration
    input_size = nNodes
    hidden_size = nNodes

    # Instance of the LSTM cell
    lstm_cell = LSTMCell(input_size, hidden_size)

    h_n, c_n = lstm_cell(xinput, pre_c, pre_h)
    return h_n, c_n


def inf_full_LSTM(input_layer, nNodes, supp_A, layer_n, lambda1, regularizer, num_hidden, pre_c, pre_h,
                  forget_bias=1.0):
    # define the full block at time t=layer_n
    # lambda1 is the coefficient of regularizer loss
    # if lambda1=None, we will not count the regularizer loss. eg. in validation and test
    device = "mps"

    in_dimension = nNodes
    out_dimension = nNodes
    AdjMatrix_tensor = torch.from_numpy(supp_A)


    weightA_init = torch.clip(torch.normal(mean=0.0, std=0.00001, size=[in_dimension, out_dimension]), 0, np.infty)
    weightA = torch.tensor(weightA_init, requires_grad=True, device=device, dtype=torch.float)
    weightA_sparse = torch.multiply(weightA, AdjMatrix_tensor)
    #if layer_n == 1 and lambda1 != None:
    #    tf.compat.v1.add_to_collection('LOSSES_COLLECTION',
    #                                   tf.keras.regularizers.l1(lambda1)(weightA_sparse))
    temp1 = torch.matmul(input_layer, weightA_sparse)
    temp1_reshape = torch.reshape(temp1, [-1, 1, nNodes])
    diag = torch.matrix_diag(input_layer)
    temp2_reshape = torch.matmul(temp1_reshape, diag)
    temp2 = torch.reshape(temp2_reshape, shape=[-1, nNodes])
    phi = input_layer + dt * (temp1 - temp2)

    cur_c, cur_h = inf_LSTM_MTM(input_layer, nNodes, pre_c, pre_h, regularizer, forget_bias=forget_bias)

    betaCNN_init = torch.ones(size=[1])*0.1
    betaCNN = torch.tensor(betaCNN_init, requires_grad=True, device=device, dtype=torch.float)

    new_inp = input_layer + betaCNN * cur_h

    error = inf_erro_CNN(new_inp, nNodes, regularizer)

    layer = torch.clip(phi - dt * error, 0, 1)

    return layer, cur_c, cur_h



def obj_func():
    lossmain = 0
    totalmae = 0
    for layer_n in range(1, n_layers):
        cur_layer, pre_c, pre_h = inf_full_LSTM(cur_layer, self.nNodes, self.supp_A, layer_n, lambda1, regularizer,
                                                num_hidden, pre_c, pre_h)
        if losstype == 'MSE':
            loss_temp = tf.reduce_mean(tf.keras.losses.MSE(y_[:, layer_n, :], cur_layer))
        if losstype == 'cs':
            loss_temp = binary_cross_entropy(y_[:, layer_n, :], cur_layer)
        totalmae += mae(y_[:, layer_n, :], cur_layer)
        lossmain = lossmain + loss_temp


dt = 1
n_layers = 21
lambda1=0.001#regularization rate for weight_A
REGULARIZATION_RATE=0.0001#regularization rate for other parameters
learning_rate=0.001


# define the input and output sizes
input_size = 1
output_size = 1
hidden_size = 64
# create the model and optimizer
model = RNN(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train the model
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()