hparams = {
    'file_name':
        'runs_all_pomdps_mi_pi_l2.txt',
    'args': [{
        'algo': 'mi',
        'spec': [
            'tiger-alt-start', 'network',
            'tmaze_5_two_thirds_up', 'example_7', '4x3.95', 'cheese.95',
            'shuttle.95',
            'paint.95'
            # 'bridge-repair',
            # 'hallway'
        ],
        'policy_optim_alg': 'pi',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'method': 'a',
        'mi_steps': 100000,
        'pi_steps': 100000,
        'lr': 1,
        'use_memory': 0,
        'use_grad': 'm',
        # 'n_mem_states': [4, 6, 8],
        'n_mem_states': 2,
        'mi_iterations': 2,
        'seed': [2020 + i for i in range(10)],
    }]
}
