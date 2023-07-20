from pathlib import Path
from popgym.envs import ALL_EASY

exp_name = Path(__file__).stem

hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': '-m grl.run_sample_based',
    'args': [
        {
            # Env
            'spec': [v['id'] for k, v in ALL_EASY.items()],
            'no_gamma_terminal': True,
            'max_episode_steps': 1000,
            'gamma': 0.99,

            # Agent
            'algo': 'multihead_rnn',
            'epsilon': 0.1,
            'arch': 'gru',
            'lr': [10**-i for i in range(3, 7)],
            # 'lr': 1e-4,
            'optimizer': 'adam',
            'feature_encoding': 'none',

            # RNN
            'hidden_size': 256,
            'value_head_layers': 0,
            'trunc': -1,
            'action_cond': 'none',

            # Multihead RNN/Lambda discrep
            'multihead_action_mode': ['mc'],
            'multihead_loss_mode': ['both', 'mc'],
            # 'multihead_loss_mode': 'both',
            'multihead_lambda_coeff': [0., 1.],
            # 'multihead_lambda_coeff': 0.,
            # 'normalize_rewards': True,

            # Replay
            'replay_size': -1,
            # 'batch_size': 16,

            # Logging and Checkpointing
            'offline_eval_freq': int(1e4),
            'offline_eval_episodes': 10,
            'offline_eval_epsilon': None, # Defaults to epsilon
            'checkpoint_freq': -1, # only save last agent

            # Experiment
            'total_steps': int(15e6),
            'seed': [2020 + i for i in range(3)],
            # 'seed': 2020,
            'study_name': exp_name
        },
    ],
    # exclusion criteria. If any of the runs match any of the
    # cross-product of all the values in the dictionary, skip
    'exclude': {
        'multihead_loss_mode': ['mc'],
        'multihead_lambda_coeff': [-1, 1]
    }
}
