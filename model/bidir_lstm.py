"""
Implementation of bidirectional RNN module for BIMODAL implementation
"""
import torch
import torch.nn as nn
import numpy as np


class BiDirLSTM(nn.Module):

    def __init__(self, input_dim=110, hidden_dim=256, layers=2):
        super(BiDirLSTM, self).__init__()

        # Dimensions
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = input_dim

        # Number of LSTM layers
        self._layers = layers

        # LSTM for forward and backward direction
        self._blstm = nn.LSTM(input_size=self._input_dim, hidden_size=self._hidden_dim, num_layers=layers,
                              dropout=0.3, bidirectional=True)

        # All weights initialized with xavier uniform
        nn.init.xavier_uniform_(self._blstm.weight_ih_l0)
        nn.init.xavier_uniform_(self._blstm.weight_ih_l1)
        nn.init.orthogonal_(self._blstm.weight_hh_l0)
        nn.init.orthogonal_(self._blstm.weight_hh_l1)

        # Bias initialized with zeros expect the bias of the forget gate
        self._blstm.bias_ih_l0.data.fill_(0.0)
        self._blstm.bias_ih_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._blstm.bias_ih_l1.data.fill_(0.0)
        self._blstm.bias_ih_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._blstm.bias_hh_l0.data.fill_(0.0)
        self._blstm.bias_hh_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._blstm.bias_hh_l1.data.fill_(0.0)
        self._blstm.bias_hh_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        # Batch normalization (Weights initialized with one and bias with zero)
        self._norm_0 = nn.LayerNorm(self._input_dim, eps=.001)
        self._norm_1 = nn.LayerNorm(2 * self._hidden_dim, eps=.001)

        # Separate linear model for forward and backward computation
        self._wpred = nn.Linear(2 * self._hidden_dim, self._output_dim)
        nn.init.xavier_uniform_(self._wpred.weight)
        self._wpred.bias.data.fill_(0.0)

    def _init_hidden(self, batch_size, device):
        '''Initialize hidden states
        :param batch_size:   size of the new batch
               device:       device where new tensor is allocated
        :return: new hidden state
        '''

        return (torch.zeros(2 * self._layers, batch_size, self._hidden_dim).to(device),
                torch.zeros(2 * self._layers, batch_size, self._hidden_dim).to(device))

    def new_sequence(self, batch_size=1, device="cpu"):
        '''Prepare model for a new sequence
        :param batch_size:   size of the new batch
               device:       device where new tensor should be allocated
        :return:
        '''
        self._hidden = self._init_hidden(batch_size, device)
        return

    def check_gradients(self):
        '''Print gradients'''
        print('Gradients Check')
        for p in self.parameters():
            print('1:', p.grad.shape)
            print('2:', p.grad.data.norm(2))
            print('3:', p.grad.data)

    def forward(self, input, next_prediction='right', device="cpu"):
        '''Forward computation
        :param input:  tensor (sequence length, batch size, encoding size)
        :param next_prediction:    new token is predicted for the left or right side of existing sequence
        :param device:  device where computation is executed
        :return pred:   prediction (batch site, encoding size)
        '''

        # If next prediction is appended at the left side, the sequence is inverted such that
        # forward and backward LSTM always read the sequence along the forward and backward direction, respectively.
        if next_prediction == 'left':
            # Reverse copy of numpy array of given tensor
            input = np.flip(input.cpu().numpy(), 0).copy()
            input = torch.from_numpy(input).to(device)

        # Normalization over encoding dimension
        norm_0 = self._norm_0(input)

        # Compute LSTM unit
        out, self._hidden = self._blstm(norm_0, self._hidden)

        # out (sequence length, batch_size, 2 * hidden dim)
        # Get last prediction from forward (0:hidden_dim) and backward direction (hidden_dim:2*hidden_dim)
        for_out = out[-1, :, 0:self._hidden_dim]
        back_out = out[0, :, self._hidden_dim:]

        # Combine predictions from forward and backward direction
        bmerge = torch.cat((for_out, back_out), -1)

        # Normalization over hidden dimension
        norm_1 = self._norm_1(bmerge)

        # Linear unit forward and backward prediction
        pred = self._wpred(norm_1)

        return pred