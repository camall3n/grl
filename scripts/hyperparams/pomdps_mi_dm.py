hparams = {
    'file_name':
        'runs_pomdps_mi_dm.txt',
    'args': [{
        'algo': 'mi',
        'spec': [
            'slippery_tmaze_5_two_thirds_up',
            'example_7',
            'tiger-alt',
            '4x3.95',
            'cheese.95',
            'network',
            'shuttle.95',
            # 'bridge-repair', 'paint.95'
            # 'hallway'
        ],
        'policy_optim_alg': 'dm',
        'method': 'a',
        'lr': 1,
        'use_memory': 0,
        'use_grad': 'm',
        'mi_iterations': 2,
        'seed': [2020 + i for i in range(10)],
    }]
}
