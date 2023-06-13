from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': '-m grl.run_sample_based',
    'args': [
        {
            # Env
            'spec': [
                'tmaze_5_two_thirds_up', 'tiger-alt-start', 'cheese.95', 'network', '4x3.95',
                'shuttle.95', 'paint.95', 'hallway'
            ],
            'no_gamma_terminal': False,
            'max_episode_steps': 1000,

            # Agent
            'algo': 'multihead_rnn',
            'epsilon': 0.1,
            'arch': 'gru',
            'lr': [10**-i for i in range(2, 7)],
            'optimizer': 'adam',

            # RNN
            'hidden_size': 12,
            'value_head_layers': 0,
            'trunc': -1, # online training
            'action_cond': 'none',

            # Multihead RNN/Lambda discrep
            'multihead_action_mode': ['td'],
            'multihead_loss_mode': ['both', 'td'],
            'multihead_lambda_coeff': [-1, 0., 1.],
            'normalize_rewards': True,

            # Replay
            'replay_size': -1,
            'batch_size': 1,

            # Logging and Checkpointing
            'offline_eval_freq': 1000,
            'offline_eval_episodes': 10,
            'offline_eval_epsilon': None, # Defaults to epsilon
            'checkpoint_freq': -1, # only save last agent

            # Experiment
            'total_steps': 150000,
            'seed': [2020 + i for i in range(5)],
            'study_name': exp_name
        },
    ],
    # exclusion criteria. If any of the runs match any of the
    # cross-product of all the values in the dictionary, skip
    'exclude': {
        'multihead_loss_mode': ['td'],
        'multihead_lambda_coeff': [-1, 1]
    }
}
