import numpy as np

hparams = {
    'file_name':
        'runs_tmaze_pe_sweep_eps.txt',
    'args': [{
        'algo': 'pe',
        'spec': 'tmaze_eps_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'tmaze_junction_up_pi': 1.0,
        'epsilon': np.linspace(0, 1, num=20),
        'method': 'a',
        'use_memory': 0,
        'value_type': 'q',
        'error_type': 'l2',
        'use_grad': 'm',
        'pe_grad_steps': 50000,
        'lr': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
