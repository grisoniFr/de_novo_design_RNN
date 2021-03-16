from sample import Sampler

for model in ['BIMODAL_fixed_512', 'BIMODAL_fixed_1024',
              'BIMODAL_random_512', 'BIMODAL_random_1024', 'BIMODAL_random_512_aug_5', 'BIMODAL_random_1024_aug_5',
              'ForwardRNN_512', 'ForwardRNN_1024',
              'FBRNN_fixed_512', 'FBRNN_fixed_1024',
              'FBRNN_random_512', 'FBRNN_random_1024', 'FBRNN_random_512_aug_5', 'FBRNN_random_1024_aug_5',
              'NADE_fixed_512', 'NADE_fixed_1024',
              'NADE_random_512', 'NADE_random_1024', 'NADE_random_512_aug_5', 'NADE_random_1024_aug_5'
              ]:
    s = Sampler(model)
    s.sample( N=1000)
