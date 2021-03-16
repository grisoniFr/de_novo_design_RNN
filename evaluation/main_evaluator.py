from evaluation import Evaluator

stor_dir = '../evaluation/'
e = Evaluator(experiment_name = 'ForwardRNN')
# evaluation of training and validation losses
e.eval_training_validation(stor_dir=stor_dir)
# evaluation of sampled molecules
e.eval_molecule(stor_dir=stor_dir)