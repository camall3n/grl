import numpy as np

hparams = {
    'file_name':
        'runs_tmaze_sweep_junction_pi.txt',
    'args': [{
        'algo': 'pe',
        'spec': 'tmaze_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'epsilon': np.linspace(0, 1, num=10),
        'method': 'a',
        'use_memory': 0,
        'value_type': 'q',
        'error_type': ['l2', 'abs'],
        'use_grad': 'm',
        'pe_grad_steps': 50000,
        'lr': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
