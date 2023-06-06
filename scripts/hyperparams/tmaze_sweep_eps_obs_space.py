import numpy as np
from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'args': [{
        'spec': 'tmaze_eps_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'tmaze_junction_up_pi': 1.,
        'epsilon': np.linspace(0, 1, num=26),
        'use_memory': '0',
        'value_type': 'q',
        'lambda_0': 0.,
        'lambda_1': 1.,
        'objective': 'obs_space',
        'alpha': 1.,
        'mi_steps': 5000,
        'pi_steps': 0,
        'init_pi': 0,
        'error_type': 'l2',
        'optimizer': 'adam',
        'lr': 0.01,
        'seed': [2020 + i for i in range(10)],
    }]
}
