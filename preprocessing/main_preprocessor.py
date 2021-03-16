def preprocess_data(filename_in='../data/chembl_smiles', filename_out='', model_type='BIMODAL', starting_point='fixed',
                    invalid=True, duplicates=True, salts=True, stereochem=True, canonicalize=True, min_len=34,
                    max_len=74, augmentation=1):

    """Pre-processing of SMILES based on the user-defined parameters
        :param filename_in     path to the file containing the SMILES to pretreat (SMILES only) -- default = ChEMBL
        :param filename_out    path for file export -- default = ../data/
        :param model_type      model to be used after data preparation -- default = 'BIMODAL'
        :param starting_point  starting point for training -- default = 'fixed'
        :param invalid         if True (default), removes invalid SMILES
        :param duplicates      if True (default), removes duplicates
        :param salts           if True (default), removes salts
        :param stereochem      if True (default), removes stereochemistry
        :param canonicalize    if True (default), produces canonical SMILES
        :param max_len         maximum length of the SMILES to retain after pretreatment
        :param min_len         minimum length of the SMILES to retain after pretreatment
        :param augmentation    augmentation folds
        :return:
    FG, v1
    """

    from preprocessor import Preprocessor
    p = Preprocessor(filename_in)
    print('Pre-processing of "' + filename_in + '" started.')

    # user-defined pretreatment
    if invalid:
        p.remove_not_valid()
        print(' invalid SMILES - removed.')

    if duplicates:
        p.remove_duplicates()
        print(' duplicate SMILES - removed.')

    if salts:
        p.remove_salts()
        print(' salts - removed.')

    if stereochem:
        p.remove_stereochem()
        print(' stereochemistry - removed.')

    if canonicalize:
        p.canonicalize()
        print(' canonicalized SMILES.')

    # retains SMILES in the defined-length
    p.remove_length(min_len, max_len)

    # prepares the data based on the method type
    dataname = filename_in.split('/')[-1]
    
    if model_type == "ForwardRNN":
        name = model_type
    else:
        name = model_type + "_" + starting_point
        if augmentation > 1 and starting_point is 'fixed':  # removes augmentation for fixed starting point
            augmentation = 1

    p.preprocess(name, aug=augmentation, length=max_len)

    if filename_out is '':
        filename_out = '../data/' + dataname + '_' + name + '.csv'

    # Store new file
    p.save_data(filename_out)
    
    print('Data processed saved')