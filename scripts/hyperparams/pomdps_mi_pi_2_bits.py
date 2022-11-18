hparams = {
    'file_name': 'runs_pomdps_mi_pi_2_bits.txt',
    'args': [
        {
            'algo': 'mi',
            'spec': [
                'slippery_tmaze_5_two_thirds_up', 'example_7', 'tiger',
                '4x3.95', 'cheese.95',
                'network', 'paint.95', 'shuttle.95',
                # 'bridge-repair'
                # 'hallway'
            ],
            'policy_optim_alg': 'pi',
            'method': 'a',
            'lr': 1,
            'use_memory': 0,
            'n_mem_states': 3,
            'use_grad': 'm',
            'mi_iterations': 2,
            'seed': [2020 + i for i in range(30)]
        }
    ]
}