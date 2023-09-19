from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'args': [{
        'spec': [
            'tiger-alt-start', 'network', 'tmaze_5_two_thirds_up', 'example_7', '4x3.95',
            'cheese.95', 'network', 'shuttle.95', 'paint.95'
            # 'hallway'
            # 'bridge-repair',
        ],
        'policy_optim_alg': 'policy_iter',
        'value_type': 'q',
        'error_type': 'l2',
        'method': 'a',
        'mi_steps': 400000,
        'pi_steps': 400000,
        'lr': 1,
        'use_memory': 0,
        # 'n_mem_states': [4, 6, 8],
        'n_mem_states': 8,
        'mi_iterations': 2,
        'seed': [2020 + i for i in range(10)],
    }]
}
