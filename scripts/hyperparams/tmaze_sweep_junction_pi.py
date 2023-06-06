import numpy as np

hparams = {
    'file_name':
        'runs_tmaze_sweep_junction_pi.txt',
    'args': [{
        'spec': 'tmaze_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'tmaze_junction_up_pi': np.linspace(0, 0.5, num=26),
        'value_type': 'q',
        'alpha': 0.,
        'use_memory': 0,
        'mi_steps': 50000,
        'pi_steps': 0,
        'init_pi': 0,
        'error_type': 'l2',
        'lr': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
