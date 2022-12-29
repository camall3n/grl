hparams = {
    'file_name':
    'runs_slippery_tmaze_two_thirds_up_mem_grad.txt',
    'args': [{
        'algo': 'pe',
        'spec': 'slippery_tmaze_5_two_thirds_up',
        'method': 'a',
        'use_memory': 0,
        'use_grad': 'm',
        'lr': 1,
        'seed': [2020 + i for i in range(30)]
    }]
}
