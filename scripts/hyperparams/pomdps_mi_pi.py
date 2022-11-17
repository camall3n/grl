hparams = {
    'file_name': 'runs_pomdps_mi_pi.txt',
    'args': [
        {
            'algo': 'mi',
            'spec': [
                # 'slippery_tmaze_5_two_thirds_up', 'example_7', 'tiger',
                # '4x3.95', 'cheese.95',
                'hallway', 'network', 'paint.95', 'shuttle.95'
            ],
            'policy_optim_alg': 'pi',
            'method': 'a',
            'lr': 1,
            'use_memory': 0,
            'use_grad': 'm',
            'mi_iterations': 2,
            'seed': [2020 + i for i in range(30)]
        }
    ]
}