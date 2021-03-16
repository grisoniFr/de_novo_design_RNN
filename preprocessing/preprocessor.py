"""
Implementation of all preprocessing steps
"""

import pandas as pd
import numpy as np
from rdkit import Chem
import sys
import os

np.random.seed(1)


class Preprocessor:

    def __init__(self, name):
        # where name is the name of the file 
        
        # List to store data
        self._data = []

        # If True, check after each function that all duplicates are still removed
        self._duplicates_removed = False

        if os.path.isfile(name + '.csv'):
            self._data = pd.read_csv(name + '.csv', header=None).values[:, 0]
        elif os.path.isfile(dname + '.tar.xz'):
            # Skip first line since empty and last line since nan
            self._data = pd.read_csv(data_name + '.tar.xz', compression='xz', header=None).values[1:-1, 0]

        # Remove empty dimensions
        self._data = np.squeeze(self._data)
        return

    def preprocess(self, name, aug=1, length=74):
        """
        Preprocess data depending on model type
        :param name:    Name of the model
        :param aug:     Data augmentation
        :return:
        """
        
        if name == "ForwardRNN":
            self.add_ending('E')
            self.add_sentinel('G')
            self.padding_right('A', l=length+2)

        elif name == "FBRNN_fixed" or name == "BIMODAL_fixed":
            self.add_middle('G')
            self.add_ending('E')
            self.add_sentinel('E')
            self.padding_left_right('A', l=length+3)

        elif name == "FBRNN_random" or name == "BIMODAL_random":
            self.add_ending('E')
            self.add_sentinel('E')
            self.add_token_random_padding(start_token='G', pad_token='A', aug=aug, l=3+length*2)

        elif name == "NADE_fixed":
            p.padding_left_right('A', l=length)
            p.add_ending('G')
            p.add_sentinel('G')

        elif name == "NADE_random":
            self.padding_left_right('A', l=length)
            self.add_ending('G')
            self.add_sentinel('G')
            self.insert_missing_token(missing_token='M', aug=aug)

        else:
            print("CAN NOT FIND MODEL")
            sys.exit()

    def remove_not_valid(self):
        """Remove all SMILES not accepted by the RDKit
        :return:
        """
        # Store index to delete
        to_delete = []

        # Find not valid SMILES
        for i, s in enumerate(self._data):
            mol = Chem.MolFromSmiles(str(s))
            if mol is None:
                to_delete.append(i)

        # Delete SMILES
        if len(to_delete) != 0:
            self._data = np.delete(self._data, to_delete)
        return

    def remove_duplicates(self):
        """Remove all SMILES appearing more than once
        :return:
        """
        self._data = np.unique(self._data)

        # Set flag to always remove duplicated after an operation
        self._duplicates_removed = True
        return

    def remove_stereochem(self):
        """Remove all token related stereochemistry
        :return:
        """
        # Token used for stereochemistry
        stereochem_token = ['/', '@', '\\']

        for t in stereochem_token:
            self.remove_token(t)

        # Remove possible created duplicates
        if self._duplicates_removed:
            self.remove_duplicates()
        return

    def remove_token(self, t):
        """Remove token t from all elements of data
        :param t:   token to remove
        :return:
        """
        self._data = np.array([d.replace(t, '') for d in self._data])

        # Remove possible created duplicates
        if self._duplicates_removed:
            self.remove_duplicates()
        return

    def remove_salts(self):
        """Remove all salts
        Non-bonded interactions are represented by '.'
        We assume that the one with the largest SMILES sequence should be preserved
        :return:
        """
        for i, s in enumerate(self._data):
            splits = s.split('.')
            # Select longest part of SMILES
            self._data[i] = max(splits, key=len)

        # Remove possible deposits
        self.remove_token('.')
        # Remove possible created duplicates
        if self._duplicates_removed:
            self.remove_duplicates()
        return

    def canonicalize(self):
        """Canonicalize all SMILES from data
        :return:
        """
        for i, s in enumerate(self._data):
            mol = Chem.MolFromSmiles(str(s))
            self._data[i] = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        # Remove possible created duplicates
        if self._duplicates_removed:
            self.remove_duplicates()
        return

    def remove_length(self, min_len=34, max_len=74):
        """Keep only SMILES with a length between min and max
        :param  min_len:    minimal length (-1: no minimal length)
                max_len:    maximal length (-1: no maximal length)
        :return:
        """
        # Store index to delete
        to_delete = []

        # Find strings longer than max
        if max_len != -1:
            for i, s in enumerate(self._data):
                if len(s) > max_len:
                    to_delete.append(i)

        # Find Strings shorter than min
        if min != -1:
            for i, s in enumerate(self._data):
                if len(s) < min_len:
                    to_delete.append(i)

        # Remove elements
        self._data = np.delete(self._data, to_delete)
        return

    def add_sentinel(self, token='E'):
        """Add token at the beginning of each SMILES
        :param  token:  token to insert
        :return:
        """
        for i, s in enumerate(self._data):
            self._data[i] = token + s
        return

    def add_ending(self, token='E'):
        """Add token at the end of each SMILES
        :param  token:  token to insert
        :return:
        """
        for i, s in enumerate(self._data):
            self._data[i] = s + token
        return

    def add_middle(self, token='G'):
        """Add token in the middle of each SMILES
        :param  token:  token to insert
        :return:
        """
        for i, s in enumerate(self._data):
            mid = len(s) // 2
            self._data[i] = s[:mid] + token + s[mid:]
        return

    def add_token_random_padding(self, start_token='G', pad_token='A', aug=5, l=0):
        '''Add start_token a n different random position and pad to have start_token in the middle of the obtained sequence
        Meathod should be applied after add_ending
        :param start_token:     token introduced in the string
        :param pad_token:       token used for padding
        :param n:               number for data augmentation
        :param l:               length of the final string (if l=0 use length of longest string)
        '''

        # Compute length of longest string
        if l == 0:
            max_l = len(max(self._data, key=len)) - 1
        else:
            max_l = l // 2

        aug_data = np.empty((self._data.size, aug)).astype(object)
        for i, s in enumerate(self._data):
            l = len(s)
            # Choose n different position for starting token (after 0 and before l-1,
            # since 0 and l-1 are special tokens for the ending (E))
            r = np.random.choice(np.arange(l - 1) + 1, aug, replace=False)

            # Tmp array to store augmentation of a SMILES
            for j, r_j in enumerate(r):
                # Added token should be located within the molecule (after 0 and before l-1,
                # since 0 and l-1 are special tokens for the ending (E)
                aug_data[i, j] = s[:r_j].rjust(max_l, pad_token) + start_token + s[r_j:].ljust(max_l, pad_token)

        # Convert array to shape (n_samples, n_augmentation)
        print(self._data.shape)
        self._data = aug_data.astype(str)

    def insert_missing_token(self, missing_token='M', aug=1):
        """Insert missing_token at random position and store changed and reference SMILES
        :param missing_token:   Token used to indicate missing value
        """

        # New data array (n_samples, 2) stores correct SMILES and SMILES with missing values
        data = np.empty((self._data.size, aug + 1)).astype(object)
        data[:, 0] = self._data
        for a in range(aug):
            data[:, a + 1] = np.copy(self._data)

        # Iteration over complete data
        for i, s in enumerate(self._data):
            # Compute length of current SMILES
            l = len(s)

            # Compute number of missing values between 0 and l-2 (First and last token are not replaced)
            n_missing = np.random.choice(np.arange(l - 2), aug, replace=False)

            for a in range(aug):
                # Compute position of missing values between 1 and l-2 (First token (0) and
                # last token (l-1) are not replaced)
                r = np.random.choice(np.arange(l - 2) + 1, n_missing[a], replace=False)

                # Insert missing values
                for r_i in r:
                    data[i, a + 1] = data[i, a + 1][:r_i] + missing_token + data[i, a + 1][r_i + 1:]

        self._data = data.astype(str)

    def padding_right(self, token='A', l=0):
        """Padding of data on the right side to obtain a consistent length
        :param token:   token used for padding
        :return l:      length of the padding (if l=0 use length of longest string)
        """
        # Compute length of longest string if no length specified
        if l == 0:
            l = len(max(self._data, key=len))

        # Padding of all strings in array
        for i, s in enumerate(self._data):
            self._data[i] = s.ljust(l, token)
        return l

    def padding_left_right(self, token='A', l=0):
        """Padding of data on the right and left side to obtain a consistent length
        :param token:   token used for padding
        :return l:      length of the padding (if l=0 use length of longest string)
        """
        # Compute length of longest string
        if l == 0:
            l = len(max(self._data, key=len))

        # Padding of all strings in array
        for i, s in enumerate(self._data):
            self._data[i] = s.center(l, token)
        return l

    def save_data(self, name='data.csv'):
        pd.DataFrame(self._data).to_csv(name, header=None, index=None)
        return

    def get_data(self):
        return self._data
