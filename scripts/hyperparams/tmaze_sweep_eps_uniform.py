import numpy as np

hparams = {
    'file_name':
        'runs_tmaze_sweep_eps.txt',
    'args': [{
        'algo': 'mi',
        'spec': 'tmaze_eps_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'tmaze_junction_up_pi': 1.,
        'epsilon': np.linspace(0, 1, num=26),
        'method': 'a',
        'use_memory': 0,
        'value_type': 'q',
        'alpha': 1.,
        'mi_steps': 50000,
        'pi_steps': 0,
        'init_pi': 0,
        'error_type': 'l2',
        'use_grad': 'm',
        'lr': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
