"""
Implementation of different training methods
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit
import pandas as pd
import configparser
from fb_rnn import FBRNN
from forward_rnn import ForwardRNN
from nade import NADE
from bimodal import BIMODAL
from one_hot_encoder import SMILESEncoder
from sklearn.utils import shuffle
import os
from helper import clean_molecule, check_model, check_molecules

np.random.seed(1)


class Trainer():

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
        self._n_folds = int(self._config['TRAINING']['n_folds'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])

        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._encoder.encode([self._config['EVALUATION']['starting_token']])


        if self._model_type == 'ForwardRNN':
            self._model = ForwardRNN(self._molecular_size, self._encoding_size,
                                     self._learning_rate, self._hidden_units)

        elif self._model_type == 'BIMODAL':
            self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units)

        self._data = self._encoder.encode_from_file(self._file_name)

    def complete_run(self, stor_dir='../evaluation/', restart=False):
        '''Training without validation on complete data'''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Build model
        self._model.build()

        # Store total Statistics
        tot_stat = []

        # only single fold
        fold = 1

        # Shuffle data before training (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
        # to  (all_SMILES, molecular_size, encoding_size))
        self._data, label = shuffle(self._data.reshape(-1, self._molecular_size, self._encoding_size),
                                    label.reshape(-1, self._molecular_size))

        for i in range(self._epochs):
            print('Fold:', fold)
            print('Epoch:', i)

            # With restart read existing files
            if restart:
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
                    continue

                # Continue with normal training
                else:
                    restart = False

            # Train model
            statistic = self._model.train(self._data, label, epochs=1, batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Store model
            self._model.save(
                stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i))

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

    def single_run(self, stor_dir='../evaluation/', restart=False):
        '''Training with validation and store data'''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/validation'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/validation')

        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Special preprocessing in the case of NADE
        if (self._model_type == 'NADE' or self._model_type == 'NADE_v2') and self._generation == 'random':
            # First column stores correct SMILES and second column stores SMILES with missing values
            label = np.argmax(self._data[:, 0], axis=-1).astype(int)
            aug = self._data.shape[1] - 1
            label = np.repeat(label[:, np.newaxis, :], aug, axis=1)
            self._data = self._data[:, 1:]

        # Split data into train and test data
        train_data, test_data, train_label, test_label = train_test_split(self._data, label, test_size=1. / 5,
                                                                          random_state=1, shuffle=True)
        # Build model
        self._model.build()

        # Store total Statistics
        tot_stat = []

        # Store validation loss
        tot_loss = []

        # only single fold
        fold = 1

        for i in range(self._epochs):
            print('Fold:', fold)
            print('Epoch:', i)

            if restart:
                # Read existing files
                tmp_val_file = pd.read_csv(
                    stor_dir + '/' + self._experiment_name + '/validation/val_fold_' + str(fold) + '.csv',
                    header=None).to_numpy()
                tmp_stat_file = pd.read_csv(
                    stor_dir + '/' + self._experiment_name + '/statistic/stat_fold_' + str(fold) + '.csv',
                    header=None).to_numpy()

                # Check if current epoch is successfully completed else continue with normal training
                if check_model(self._model_type, self._experiment_name, stor_dir, fold, i) and check_molecules(
                        self._experiment_name, stor_dir, fold, i) and tmp_val_file.shape[0] > i and tmp_stat_file.shape[
                    0] > i:

                    # Load model
                    self._model.build(
                        stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i))

                    # Fill statistic and loss list
                    tot_stat.append(tmp_stat_file[i, 1:].reshape(1, -1).tolist())
                    tot_loss.append(tmp_val_file[i, 1])

                    # Skip this epoch
                    continue

                # Continue with normal training
                else:
                    restart = False

            # Train model (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
            # to  (all_SMILES, molecular_size, encoding_size))
            statistic = self._model.train(train_data.reshape(-1, self._molecular_size, self._encoding_size),
                                          train_label.reshape(-1, self._molecular_size), epochs=1,
                                          batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Store model
            self._model.save(
                stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i))

            # Test model on validation set
            tot_loss.append(
                self._model.validate(test_data.reshape(-1, self._molecular_size, self._encoding_size),
                                     test_label.reshape(-1, self._molecular_size)))

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

            # Store validation data
            pd.DataFrame(np.array(tot_loss).reshape(-1, 1)).to_csv(
                stor_dir + '/' + self._experiment_name + '/validation/val_fold_' + str(fold) + '.csv',
                header=None)

    def cross_validation(self, stor_dir='../evaluation/', restart=False):
        '''Perform cross-validation and store data'''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/validation'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/validation')

        self._kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=2)

        # Count iterations
        fold = 0

        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Split data into train and test data
        for train, test in self._kf.split(self._data):

            # Shuffle index within test and train set
            np.random.shuffle(train)
            np.random.shuffle(test)

            fold += 1

            self._model.build()

            # Store total statistics
            tot_stat = []

            # Store validation loss
            tot_loss = []

            for i in range(self._epochs):
                print('Fold:', fold)
                print('Epoch:', i)

                if restart:
                    tmp_val_file = pd.read_csv(
                        stor_dir + '/' + self._experiment_name + '/validation/val_fold_' + str(fold) + '.csv',
                        header=None).to_numpy()

                    tmp_stat_file = pd.read_csv(
                        stor_dir + '/' + self._experiment_name + '/statistic/stat_fold_' + str(fold) + '.csv',
                        header=None).to_numpy()

                    # Check if current epoch is successfully complete[0]d else continue with normal training
                    if check_model(self._model_type, self._experiment_name, stor_dir, fold, i) and check_molecules(
                            self._experiment_name, stor_dir, fold, i) and tmp_val_file.shape[0] > i and tmp_stat_file.shape[
                        0] > i:

                        # Load model
                        self._model.build(
                            stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i))

                        # Fill statistic and loss list
                        tot_stat.append(tmp_stat_file[i, 1:].reshape(1, -1).tolist())
                        tot_loss.append(tmp_val_file[i, 1])

                        # Skip this epoch
                        continue

                    else:
                        restart = False

                # Train model (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
                # to  (all_SMILES, molecular_size, encoding_size))
                statistic = self._model.train(
                    self._data[train].reshape(-1, self._molecular_size, self._encoding_size),
                    label[train].reshape(-1, self._molecular_size), epochs=1, batch_size=self._batch_size)

                tot_stat.append(statistic.tolist())

                # Store model
                self._model.save(
                    stor_dir + '/' + self._experiment_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(i))

                # Test model on validation set
                tot_loss.append(
                    self._model.validate(self._data[test].reshape(-1, self._molecular_size, self._encoding_size),
                                         label[test].reshape(-1, self._molecular_size)))

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
                    stor_dir + '/' + self._experiment_name + '/statistic/stat_fold_' + str(fold) + '.csv', header=None)

                # Store validation data
                pd.DataFrame(np.array(tot_loss).reshape(-1, 1)).to_csv(
                    stor_dir + '/' + self._experiment_name + '/validation/val_fold_' + str(fold) + '.csv', header=None)


