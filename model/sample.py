"""
Implementation of the sampler to generate SMILES from a trained model

"""

import pandas as pd
import numpy as np
import configparser
from forward_rnn import ForwardRNN
from one_hot_encoder import SMILESEncoder
from bimodal import BIMODAL
import os
from helper import clean_molecule, check_valid

np.random.seed(1)


class Sampler():

    def __init__(self, experiment_name):
        self._encoder = SMILESEncoder()

        # Read parameter used during training
        self._config = configparser.ConfigParser()
        self._config.read('../experiments/' + experiment_name + '.ini')
        
        self._model_type = self._config['MODEL']['model']
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])

        self._file_name = self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])

        self._epochs = int(self._config['TRAINING']['epochs'])
        self._n_folds = int(self._config['TRAINING']['n_folds'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])

        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._encoder.encode([self._config['EVALUATION']['starting_token']])

        if self._model_type == 'FBRNN':
            self._model = FBRNN(self._molecular_size, self._encoding_size,
                                self._learning_rate, self._hidden_units)
        elif self._model_type == 'ForwardRNN':
            self._model = ForwardRNN(self._molecular_size, self._encoding_size,
                                     self._learning_rate, self._hidden_units)

        elif self._model_type == 'BIMODAL':
            self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units)

        elif self._model_type == 'NADE':
            self._generation = self._config['MODEL']['generation']
            self._missing_token = self._encoder.encode([self._config['TRAINING']['missing_token']])
            self._model = NADE(self._molecular_size, self._encoding_size, self._learning_rate,
                               self._hidden_units, self._generation, self._missing_token)

        # Read data
        if os.path.isfile('../data/' + self._file_name + '.csv'):
            self._data = pd.read_csv('../data/' + self._file_name + '.csv', header=None).values[:, 0]
        elif os.path.isfile('../data/' + self._file_name + '.tar.xz'):
            # Skip first line since empty and last line since nan
            self._data = pd.read_csv('../data/' + self._file_name + '.tar.xz', compression='xz', header=None).values[
                         1:-1, 0]

        # Clean data from start, end and padding token
        for i, mol_dat in enumerate(self._data):
            self._data[i] = clean_molecule(mol_dat, self._model_type)

    def sample(self, N=100, stor_dir='../evaluation', T=0.7, fold=[1], epoch=[9], valid=True, novel=True, unique=True, write_csv=True):

        '''Sample from a model where the number of novel valid unique molecules is fixed
        :param stor_dir:    directory where the generated SMILES are saved
        :param N:        number of samples
        :param T:        Temperature
        :param fold:     Folds to use for sampling
        :param epoch:    Epochs to use for sampling
        :param valid:    If True, only accept valid SMILES
        :param novel:    If True, only accept novel SMILES
        :param unique:   If True, only accept unique SMILES
        :param write_csv If True, the generated SMILES are written in stor_dir
        :return: res_molecules: list with all the generated SMILES
        '''
        
        res_molecules = []
        print('Sampling: started')
        for f in fold:
            for e in epoch:
                self._model.build(
                    stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(f) + '_epochs_' + str(e))

                new_molecules = []
                while len(new_molecules) < N:
                    new_mol = self._encoder.decode(self._model.sample(self._starting_token, T))

                    # Remove remains from generation
                    new_mol = clean_molecule(new_mol[0], self._model_type)

                    # If not valid, get new molecule
                    if valid and not check_valid(new_mol):
                        continue

                    # If not unique, get new molecule
                    if unique and (new_mol in new_molecules):
                        continue

                    # If not novel, get molecule
                    if novel and (new_mol in self._data):
                        continue

                    # If all conditions checked, add new molecule
                    new_molecules.append(new_mol)

                # Prepare name for file
                name = 'molecules_fold_' + str(f) + '_epochs_' + str(e) + '_T_' + str(T) + '_N_' + str(N) + '.csv'
                if unique:
                    name = 'unique_' + name
                if valid:
                    name = 'valid_' + name
                if novel:
                    name = 'novel_' + name

                # Store final molecules
                if write_csv:
                    if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules/'):
                        os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules/')
                    mol = np.array(new_molecules).reshape(-1)
                    pd.DataFrame(mol).to_csv(stor_dir + '/' + self._experiment_name + '/molecules/' + name, header=None)
        
            res_molecules.append(new_molecules)
        
        print('Sampling: done')
        return res_molecules
        
