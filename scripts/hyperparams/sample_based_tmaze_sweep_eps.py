import numpy as np
from pathlib import Path

exp_name = Path(__file__).stem
hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'scripts/learning_agent/memory_iteration.py',
    'args': [{
        'env': 'tmaze_5_two_thirds_up',
        'study_name': exp_name,
        'load_policy': True,
        'policy_junction_up_prob': 1.,
        'policy_epsilon': np.linspace(0, 1, num=26),
        'learning_rate': 0.001,
        'trial_id': [i for i in range(1, 11)],
    }]
}
