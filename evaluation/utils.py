# utilities to generate molecules with this repo.
# v1. Dec 2020, F. Grisoni

def make_config(model_type='BIMODAL', net_size=512, epochs=10, starting_point='random', fine_tuning='fine_tuning', n_sampling=1000, T_sampling=0.7,augmentation_level=1):
    # writes the configuration file for fine-tuning depending on user-defined settings
    # model (str): 'BIMODAL' or 'ForwardRNN'
    # net_size (int): size of the network
    # epochs (int): fine-tuning epochs
    # start (str): 'random' or 'fixed'
    # fine_tuning (str): name of the fine-tuning set
    # n_sampling (int): molecules to sample for each fine-tuning epoch
    # T_sampling (double): sampling temperature

    import configparser

    # name of the configuration file to use
    reference_name = model_type + '_' + starting_point + '_' + str(net_size)
    if augmentation_level == 1:
        reference_name = model_type + '_' + starting_point + '_' + str(net_size)
    else:
        reference_name = model_type + '_' + starting_point + '_' + str(net_size) + '_aug_' + str(augmentation_level)

    exp_name = reference_name + '_FineTuning'

    # file to use as template
    if model_type is 'BIMODAL':
        template_name = 'BIMODAL_random_512_FineTuning_template.ini'
    else:
        template_name = 'ForwardRNN_512_FineTuning_template.ini'

    # location of processed fine tuning set
    fine_tuning_preprocessed = fine_tuning + '_' + model_type + '_' + starting_point

    # reads the config file from the template
    config = configparser.ConfigParser()
    config.sections()
    config.read('../experiments/' + template_name)  # starts from one of the templates

    # changes the fields based on the specified options
    config['MODEL']['model'] = model_type
    if model_type is 'BIMODAL':
        config['MODEL']['hidden_units'] = str(net_size//4)
    else:
        config['MODEL']['hidden_units'] = str(net_size // 2)

    # start writing the config file
    config['DATA']['data'] = fine_tuning_preprocessed
    config['TRAINING']['epochs'] = str(epochs)

    config['EVALUATION']['samples'] = str(n_sampling)
    config['EVALUATION']['temp'] = str(T_sampling)

    # picks one of our pre-trained models that are provided in the repo.
    # If the SMILES preprocessing changes, the pre-training has to be performed again
    config['FINETUNING']['start_model'] = '../evaluation/' + reference_name + '/models/model_fold_1_epochs_9'

    # writes back the new options
    with open('../experiments/' + exp_name + '.ini', 'w') as configfile:
        config.write(configfile)

    return exp_name

