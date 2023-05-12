import numpy as np

hparams = {
    'file_name':
        'runs_tmaze_sweep_alpha.txt',
    'args': [{
        'algo': 'mi',
        'spec': 'tmaze_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'tmaze_junction_up_pi': 2/3,
        'epsilon': 0.,
        'method': 'a',
        'use_memory': 0,
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': np.linspace(0., 1., num=20),
        'flip_count_prob': [True, False],
        'pi_steps': 0,
        'init_pi': 0,
        'use_grad': 'm',
        'lr': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
