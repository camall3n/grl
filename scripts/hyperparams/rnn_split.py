from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry':
        '-m grl.run_sample_based',
    'args': [
        {
            # Env
            'spec': [
                'cheese.95', 'tiger-alt-start', 'network', 'tmaze_5_two_thirds_up', '4x3.95',
                'shuttle.95', 'paint.95', 'hallway'
            ],
            'no_gamma_terminal': False,
            'max_episode_steps': 1000,
            'feature_encoding': 'one_hot',

            # Agent
            'algo': 'multihead_rnn',
            'epsilon': 0.1,
            'arch': 'gru',
            'lr': [10**-i for i in range(2, 6)],
            'optimizer': 'adam',

            # RNN
            'hidden_size': 12,
            'value_head_layers': 1,
            'trunc': 10,
            'action_cond': 'cat',

            # Multihead RNN/Lambda discrep
            'multihead_action_mode': 'td',
            'multihead_loss_mode': ['both', 'split'],
            'multihead_lambda_coeff': 0,
            'normalize_rewards': True,
            'residual_obs_val_input': True,

            # Replay
            'replay_size': 1000,
            'batch_size': 8,

            # Logging and Checkpointing
            'offline_eval_freq': 1000,
            'offline_eval_episodes': 5,
            'offline_eval_epsilon': None, # Defaults to epsilon
            'checkpoint_freq': -1, # only save last agent

            # Experiment
            'total_steps': 150000,
            'seed': [2020 + i for i in range(5)],
            'study_name': exp_name
        },
    ]
}
