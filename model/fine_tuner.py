""""
Implementation of fine-tuning methods
"""

import numpy as np
import pandas as pd
import configparser
from forward_rnn import ForwardRNN
from bimodal import BIMODAL
from one_hot_encoder import SMILESEncoder
import os
from helper import clean_molecule, check_model, check_molecules

np.random.seed(1)


class FineTuner():

    def __init__(self, experiment_name='ForwardRNN'):

        self._encoder = SMILESEncoder()

        # Read all parameter from the .ini file
        self._config = configparser.ConfigParser()
        self._config.read('../experiments/' + experiment_name + '.ini')

        self._model_type = self._config['MODEL']['model']
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])

        self._file_name = '../data/' + self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])

        self._epochs = int(self._config['TRAINING']['epochs'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])

        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._encoder.encode([self._config['EVALUATION']['starting_token']])

        # Read starting model
        self._start_model = self._config['FINETUNING']['start_model']

        if self._model_type == 'ForwardRNN':
            self._model = ForwardRNN(self._molecular_size, self._encoding_size,
                                     self._learning_rate, self._hidden_units)

        elif self._model_type == 'BIMODAL':
            self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units)

        self._data = self._encoder.encode_from_file(self._file_name)

    def fine_tuning(self, stor_dir='../evaluation/', restart=False):
        '''Perform fine-tuning and store statistic,
        NOTE: Directory should be prepared with the correct name and model
        NOTE: Molecules are not generated or validation is not performed. To sample molecules sampler should be used'
        :param stor_dir:    directory to store data
        :return:
        '''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Build model
        # self._model.build(stor_dir + '/' + self._experiment_name + '/' + self._start_model)
        self._model.build(self._start_model)

        # Store total Statistics
        tot_stat = []

        # only one fold for fine-tuning
        fold = 1

        for i in range(self._epochs):
            print('Epoch:', i+1)

            if restart:
                # Read existing files
                tmp_stat_file = pd.read_csv(
                    stor_dir + '/' + self._experiment_name + '/statistic/stat_fold_' + str(fold) + '.csv',
                    header=None).to_numpy()

                # Check if current epoch is successfully completed else continue with normal training
                if check_model(self._model_type, self._experiment_name, stor_dir, fold, i) and check_molecules(
                        self._experiment_name, stor_dir, fold, i) and tmp_stat_file.shape[0] > i:
                    # Load model
                    self._model.build(
                        stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i))

                    # Fill statistic and loss list
                    tot_stat.append(tmp_stat_file[i, 1:].reshape(1, -1).tolist())

                    # Skip this epoch
                    continue

                else:
                    restart = False

            # Train model (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
            # to  (all_SMILES, molecular_size, encoding_size))
            statistic = self._model.train(self._data.reshape(-1, self._molecular_size, self._encoding_size),
                                          label.reshape(-1, self._molecular_size), epochs=1, batch_size=int(self._batch_size))
            tot_stat.append(statistic.tolist())

            # Store model
            self._model.save(
                stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i) )

            # Sample new molecules
            new_molecules = []
            for s in range(self._samples):
                mol = self._encoder.decode(self._model.sample(self._starting_token, self._T))
                new_molecules.append(clean_molecule(mol[0], self._model_type))

            # Store new molecules
            new_molecules = np.array(new_molecules)
            pd.DataFrame(new_molecules).to_csv(
                stor_dir + '/' + self._experiment_name + '/molecules/molecule_fold_' + str(fold) + '_epochs_' + str(
                    i) + '.csv', header=None)

            # Store statistic
            store_stat = np.array(tot_stat).reshape(i + 1, -1)
            pd.DataFrame(np.array(store_stat)).to_csv(
                stor_dir + '/' + self._experiment_name + '/statistic/stat_fold_' + str(fold) + '.csv',
                header=None)
