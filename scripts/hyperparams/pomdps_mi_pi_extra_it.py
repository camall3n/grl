hparams = {
    'file_name': 'runs_pomdps_mi_pi_extra_it.txt',
    'args': [
        {
            'algo': 'mi',
            'spec': [
                'tmaze_5_two_thirds_up',
                'example_7', 'tiger-alt',
                '4x3.95', 'cheese.95',
                'network', 'shuttle.95',
                'paint.95'

                # 'bridge-repair',
                # 'hallway'
            ],
            'policy_optim_alg': 'pi',
            'method': 'a',
            'mi_steps': 100000,
            'pi_steps': 100000,
            'lr': 1,
            'use_memory': 0,
            'use_grad': 'm',
            'n_mem_states': [3, 4],
            'mi_iterations': 2,
            'seed': [2020 + i for i in range(10)]
        }
    ]
}