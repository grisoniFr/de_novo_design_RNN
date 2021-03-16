"""
Implementation of one-directional RNN for SMILES generation
"""

import numpy as np
import torch
import torch.nn as nn
from one_out_lstm import OneOutLSTM
import torch.nn.functional as F
from scipy.misc import logsumexp

torch.manual_seed(1)
np.random.seed(5)


class ForwardRNN():

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=256):

        self._molecule_size = molecule_size
        self._input_dim = encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units

        # Learning rate
        self._lr = lr

        # Build new model
        self._lstm = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        # Check availability of GPUs
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def print_model(self):
        '''Print name and shape of all tensors'''
        for name, p in self._lstm.state_dict().items():
            print(name)
            print(p.shape)

    def build(self, name=None):
        """Build new model or load model by name"""
        if (name is None):
            self._lstm = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        else:
            self._lstm = torch.load(name + '.dat', map_location=self._device)

        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

    def train(self, data, label, epochs=1, batch_size=1):
        '''Train the model
        :param  data:   data array (n_samples, molecule_size, encoding_length)
        :param  label:  label array (n_samples, molecule_size)
        :param  epochs: number of epochs for training
        :param  batch_size: batch_size for training
        :return statistic:  array storing computed losses (epochs, batch)
        '''

        # Compute tensor of labels
        label = torch.from_numpy(label).to(self._device)

        # Number of samples
        n_samples = data.shape[0]

        # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size, n_samples, encoding_dim)
        data = np.swapaxes(data, 0, 1).astype('float32')

        # Create tensor
        data = torch.from_numpy(data).to(self._device)

        # Calculate number of batches per epoch
        if (n_samples % batch_size) is 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        # Store losses
        statistic = np.zeros((epochs, n_iter))

        # Prepare model
        self._lstm.train()

        # Iteration over epochs
        for i in range(epochs):

            # Iteration over batches
            for n in range(n_iter):

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Initialize loss for molecule
                molecule_loss = torch.zeros(1).to(self._device)

                # Reset model with correct batch size
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # Iteration over molecules
                for j in range(self._molecule_size - 1):
                    # Prepare input tensor with dimension (1,batch_size, encoding_dim)
                    input = data[j, batch_start:batch_end, :].view(1, batch_end - batch_start, -1)

                    # Probabilities next prediction
                    forward = self._lstm(input)

                    # Mean cross-entropy loss
                    loss_forward = self._loss(forward.view(batch_end - batch_start, -1),
                                              label[batch_start:batch_end, j + 1])

                    # Add to molecule loss
                    molecule_loss = torch.add(molecule_loss, loss_forward)

                # Compute backpropagation
                self._optimizer.zero_grad()
                molecule_loss.backward(retain_graph=True)

                # Store statistics: loss per token (middle token not included)
                statistic[i, n] = molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)

                # Perform optimization step and reset gradients
                self._optimizer.step()

        # print('Statistic:', statistic)
        return statistic

    def validate(self, data, label):
        ''' Validation of model and compute error
        :param data:    test data (n_samples, molecule_size, encoding_size)
        :return:        mean loss over test data
        '''

        # Use train mode to get loss consistent with training
        self._lstm.train()

        # Gradient is not compute to reduce memory usage
        with torch.no_grad():
            # Compute tensor of labels
            label = torch.from_numpy(label).to(self._device)

            # Number of samples
            n_samples = data.shape[0]

            # Change axes from (n_samples, molecule_size , encoding_dim) to (molecule_size , n_samples, encoding_dim)
            data = np.swapaxes(data, 0, 1).astype('float32')

            # Create tensor for data and store at correct device
            data = torch.from_numpy(data).to(self._device)

            # Initialize loss for molecule at correct device
            molecule_loss = torch.zeros(1).to(self._device)

            # Reset model with correct batch size and device
            self._lstm.new_sequence(n_samples, self._device)

            for j in range(self._molecule_size - 1):
                # Prepare input tensor with dimension (1,n_samples, 2*molecule_size)
                input = data[j, :, :].view(1, n_samples, -1)

                # Probabilities next prediction
                forward = self._lstm(input)

                # Mean cross-entropy loss
                loss_forward = self._loss(forward.view(n_samples, -1),
                                          label[:, j + 1])

                # Add to molecule loss
                molecule_loss = torch.add(molecule_loss, loss_forward)

        return molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)

    def sample(self, start_token, T=1):
        '''Generate new molecule
        :param middle_token:    starting token
        :param T:               sampling temperature
        :return molecule:       newly generated molecule (molecule_length, encoding_length
        '''
        # Prepare model
        self._lstm.eval()

        # Gradient is not compute to reduce memory usage
        with torch.no_grad():
            # Output array s
            output = np.zeros((self._molecule_size, self._input_dim))

            # Store molecule
            molecule = np.zeros((1, self._molecule_size, self._input_dim))

            # Set start token as first output
            output[0, :] = start_token[:]

            # Set start token for molecule
            molecule[0, 0, :] = start_token[:]

            # Prepare input as tensor at correct device
            input = torch.from_numpy(np.array(output[0, :]).astype(np.float32)).view(1, 1, -1).to(self._device)

            # Prepare model
            self._lstm.new_sequence(batch_size=1, device=self._device)

            # Sample from model
            for j in range(self._molecule_size - 1):
                # Compute prediction
                forward = self._lstm(input)

                # Conversion to numpy and creation of new token by sampling from the obtained probability distribution
                token_forward = self.sample_token(np.squeeze(forward.cpu().detach().numpy()), T)

                # Set selected tokens
                molecule[0, j + 1, token_forward] = 1.0

                # Prepare input of next step
                output[j + 1, token_forward] = 1.0
                input = torch.from_numpy(output[j + 1, :].astype(np.float32)).view(1, 1, -1).to(self._device)

        return molecule

    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''
        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature
        p = np.exp(out / T) / np.sum(np.exp(out / T))

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='test_model'):
        torch.save(self._lstm, name + '.dat')
