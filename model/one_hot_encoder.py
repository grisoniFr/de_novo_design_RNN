"""
Implementation of one-hot-encoder for SMILES strings
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import sys


class SMILESEncoder():

    def __init__(self):
        # Allowed tokens (adapted from default dictionary)
        self._tokens = np.sort(['#', '=',
                                '\\', '/', '%', '@', '+', '-', '.',
                                '(', ')', '[', ']',
                                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                                'A', 'B', 'E', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V',
                                'Z',
                                'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't'
                                ])

        # Dictionary mapping index to token
        self._encoder = OneHotEncoder(categories=[self._tokens], dtype=np.uint8, sparse=False)

    def encode_from_file(self, name='data'):
        '''One-hot-encoding from .csv file
        :param name:    name of data file
        :return:    encoded data (data size, molecule size, allowed token size)
        '''

        # Read data
        if os.path.isfile(name + '.csv'):
            data = pd.read_csv(name + '.csv', header=None).values
        elif os.path.isfile(name + '.tar.xz'):
            # Skip first line since empty and last line since nan
            data = pd.read_csv(name + '.tar.xz', compression='xz', header=None).values[1:-1]
        else:
            print('CAN NOT READ DATA')
            sys.exit()

        # Store dimensions
        shape = data.shape
        data = data.reshape(-1)

        # Remove empty dimensions
        data = np.squeeze(data)

        # Return array with same first and second dimensions as input
        return self.encode(data).reshape((shape[0], shape[1], -1, len(self._tokens)))

    def encode(self, data):
        '''One-hot-encoding
        :param data:         input data (sample size,)
        :return one_hot:     encoded data (sample size, molecule size, allowed token size)
        '''

        # Split SMILES into characters
        data = self.smiles_to_char(data)

        # Store dimensions and reshape to use encoder
        shape = data.shape
        data = data.reshape((-1, 1))

        # Encode SMILES
        data = self._encoder.fit_transform(data)

        # Restore shape
        data = data.reshape((shape[0], shape[1], -1))

        return data

    def decode(self, one_hot):
        '''Decode one-hot encoding to SMILES
        :param one_hot:    one_hot data (sample size, molecule size, allowed token size)
        :return data:      SMILES (sample size,)
        '''

        # Store dimensions and reshape to use encoder
        shape = one_hot.shape[0]
        one_hot = one_hot.reshape((-1, len(self._tokens)))

        # Decode SMILES
        data = self._encoder.inverse_transform(one_hot)

        # Restore shape
        data = data.reshape((shape, -1))
        # Merge char to SMILES
        smiles = self.char_to_smiles(data)

        return smiles

    def smiles_to_char(self, data):
        '''Split SMILES into array of char
        :param data:        input data (sample size,)
        :return char_data:  encoded data (sample size, molecule size)
        '''
        char_data = []
        for i, s in enumerate(data):
            char_data.append(np.array(list(s)))
        # Get array from list
        char_data = np.stack(char_data, axis=0)

        return char_data

    def char_to_smiles(self, char_data):
        '''Merge array of char into SMILES
        :param char_data:   input data (sample size, molecule size)
        :return data:       encoded data (sample size, )
        '''
        data = []
        for i in range(char_data.shape[0]):
            data.append(''.join(char_data[i, :]))

        return np.array(data)
