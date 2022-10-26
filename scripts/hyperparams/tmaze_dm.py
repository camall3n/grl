hparams = {
    'file_name': 'runs_tmaze_dm.txt',
    'args': [
        {
            'algo': 'mi',
            'spec': 'tmaze_5_two_thirds_up',
            'method': 'a',
            'use_memory': 0,
            'use_grad': 'm',
            'policy_optim_alg': 'dm',
            'mi_iterations': 0,
            'seed': [2020 + i for i in range(30)]
        }
    ]
}
