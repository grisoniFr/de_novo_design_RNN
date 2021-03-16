from fine_tuner import FineTuner

for m in ['BIMODAL_random_512_FineTuning_template']:
    t = FineTuner(m)
    t.fine_tuning(stor_dir='../evaluation/')
