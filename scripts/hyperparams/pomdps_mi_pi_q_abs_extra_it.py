hparams = {
    'file_name':
        'runs_pomdps_mi_pi_q_abs_extra_it.txt',
    'args': [{
        'algo': 'mi',
        'spec': [
            # 'tiger-alt-start', 'network'
            # 'tmaze_5_two_thirds_up', 'example_7', '4x3.95', 'cheese.95', 'network',
            # 'shuttle.95',
            'paint.95'
            # 'bridge-repair',
            # 'hallway'
        ],
        'policy_optim_alg': 'pi',
        'value_type': 'q',
        'error_type': 'abs',
        'method': 'a',
        'mi_steps': 200000,
        'pi_steps': 200000,
        'lr': 1,
        'use_memory': 0,
        'use_grad': 'm',
        # 'n_mem_states': [4, 6, 8],
        'n_mem_states': 6,
        'platform': 'gpu',
        'mi_iterations': 2,
        'seed': [2020 + i for i in range(10)],
    }]
}
