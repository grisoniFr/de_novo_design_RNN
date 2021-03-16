"""
Implementation of different function simplifying SMILES handling
"""

import numpy as np
from rdkit import Chem
import os


def clean_molecule(m, model_type):
    ''' Depending on the model different remains from generation should be removed
    :param m:   molecule with padding
    :param model_type:  Type of the model
    :return:    cleaned molecule
    '''
    if model_type == 'FBRNN':
        m = remove_right_left_padding(m)
    elif model_type == 'BIMODAL':
        m = remove_right_left_padding(m, start='G', end='E')
    else:
        print("CANNOT FIND MODEL")

    m = remove_token([m], 'G')

    return m[0]


def remove_right_left_padding(mol, start='G', end='E'):
    '''Remove right and left padding from start to end token
    :param mol:     SMILES string
    :param start:   token where to start
    :param end:     token where to finish
    :return:        new SMILES where padding is removed
    '''
    # Find start and end index
    mid_ind = mol.find(start)
    end_ind = mol.find(end, mid_ind)
    start_ind = len(mol) - mol[::-1].find(end, mid_ind) - 1
    return mol[start_ind + 1:end_ind]


def remove_right_padding(mol, end='E'):
    '''Remove right and left padding from start to end token
    :param mol:     SMILES string
    :param end:     token where to finish
    :return:        new SMILES where padding is removed'''
    end_ind = mol.find(end)
    return mol[:end_ind]


def check_valid(mol):
    '''Check if SMILES is valid
    :param mol:     SMILES string
    :return:        True / False
    '''
    # Empty SMILES
    # not accepted
    if mol == '':
        return False

    # Check valid with RDKit
    # MolFromSmiles returns None if molecule not valid
    mol = Chem.MolFromSmiles(mol, sanitize=True)
    if mol is None:
        return False
    return True


def remove_token(mol, t='G'):
    '''Remove specific token from SMILES
    :param mol: SMILES string
    :param t:   token to be removed
    :return:    new SMILES string without token t
    '''
    mol = np.array([d.replace(t, '') for d in mol])
    return mol


def check_model(model_type, model_name, stor_dir, fold, epoch):
    '''Perform fine-tuning and store statistic,
    :param stor_dir:    directory of stored data
    :param fold:    Fold to check
    :param epoch:   Epoch to check
    :return exists_model:   True if model exists otherwise False
    '''

    if model_type == 'NADE':
        exists_model = os.path.isfile(
            stor_dir + '/' + model_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(
                epoch) + 'backdir.dat') and \
                       os.path.isfile(stor_dir + '/' + model_name + '/models/model_fold_' + str(
                           fold) + '_epochs_' + str(i) + 'fordir.dat')
    else:
        exists_model = os.path.isfile(
            stor_dir + '/' + model_name + '/models/model_fold_' + str(fold) + '_epochs_' + str(
                epoch) + '.dat')

    return exists_model


def check_molecules(model_name, stor_dir, fold, epoch):
    '''Perform fine-tuning and store statistic,
    :param stor_dir:    directory of stored data
    :param fold:    Fold to check
    :param epoch:   Epoch to check
    :return :   True if molecules exist otherwise False
    '''

    return os.path.isfile(
        stor_dir + '/' + model_name + '/molecules/molecule_fold_' + str(fold) + '_epochs_' + str(
            epoch) + '.csv')
