"""
Implementation of LSTM for Forward RNN
"""
import torch
import torch.nn as nn


class OneOutLSTM(nn.Module):

    def __init__(self, input_dim=55, hidden_dim=256, layers=2):
        super(OneOutLSTM, self).__init__()

        # Dimensions
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = input_dim

        # Number of LSTM layers
        self._layers = layers

        # LSTM mod
        self._lstm = nn.LSTM(input_size=self._input_dim, hidden_size=self._hidden_dim, num_layers=layers, dropout=0.3)
        # All weights initialized with xavier uniform
        nn.init.xavier_uniform_(self._lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self._lstm.weight_ih_l1)
        nn.init.orthogonal_(self._lstm.weight_hh_l0)
        nn.init.orthogonal_(self._lstm.weight_hh_l1)

        # Bias initialized with zeros expect the bias of the forget gate
        self._lstm.bias_ih_l0.data.fill_(0.0)
        self._lstm.bias_ih_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._lstm.bias_ih_l1.data.fill_(0.0)
        self._lstm.bias_ih_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._lstm.bias_hh_l0.data.fill_(0.0)
        self._lstm.bias_hh_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._lstm.bias_hh_l1.data.fill_(0.0)
        self._lstm.bias_hh_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        # Batch normalization (Weights initialized with one and bias with zero)
        self._norm_0 = nn.LayerNorm(self._input_dim, eps=.001)
        self._norm_1 = nn.LayerNorm(self._hidden_dim, eps=.001)

        # Separate linear model for forward and backward computation
        self._wforward = nn.Linear(self._hidden_dim, self._output_dim)
        nn.init.xavier_uniform_(self._wforward.weight)
        self._wforward.bias.data.fill_(0.0)

    def _init_hidden(self, batch_size, device):
        '''Initialize hidden states
        :param batch_size:  batch size
        :param device:      device to store tensors
        :return: new hidden state
        '''

        return (torch.zeros(self._layers, batch_size, self._hidden_dim).to(device),
                torch.zeros(self._layers, batch_size, self._hidden_dim).to(device))

    def new_sequence(self, batch_size=1, device="cpu"):
        '''Prepare model for a new sequence
        :param batch_size:  batch size
        :param device:      device to store tensors
        '''
        self._hidden = self._init_hidden(batch_size, device)
        return

    def check_gradients(self):
        '''Check gradients'''
        print('Gradients Check')
        for p in self.parameters():
            print(p.grad.shape)
            print(p.grad.data.norm(2))

    def forward(self, input):
        '''Forward computation
        :param input:           tensor( sequence length, batch size, encoding size)
        :return forward:      forward prediction (batch site, encoding size)
        '''

        # Normalization over encoding dimension
        norm_0 = self._norm_0(input)

        # Compute LSTM unit
        out, self._hidden = self._lstm(norm_0, self._hidden)

        # Normalization over hidden dimension
        norm_1 = self._norm_1(out)

        # Linear unit forward prediction
        forward = self._wforward(norm_1)

        return forward
