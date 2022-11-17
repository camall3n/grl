hparams = {
    'file_name': 'runs_tmaze_mi_pi.txt',
    'args': [
        {
            'algo': 'mi',
            'spec': 'tmaze_5_two_thirds_up',
            'method': 'a',
            'use_memory': 0,
            'use_grad': 'm',
            'mi_iterations': 1,
            'seed': [2020 + i for i in range(30)]
        }
    ]
}