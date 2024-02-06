from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m grl.batch_run',
    'args': [{
        'spec': [
            'tiger-alt-start', 'tmaze_5_two_thirds_up', '4x3.95',
            'cheese.95', 'network', 'shuttle.95', 'paint.95'
            # 'hallway'
            # 'bridge-repair',
        ],
        'policy_optim_alg': 'policy_grad',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'mi_steps': 20000,
        'pi_steps': 10000,
        'optimizer': 'adam',
        'lr': 0.01,
        'n_mem_states': [2, 4],
        'mi_iterations': 1,
        'random_policies': 10,
        'n_seeds': 10,
        'platform': 'gpu'
    },
    #     {
    #     'spec': ['paint.95', 'tmaze_5_two_thirds_up'],
    #     # 'spec': [
    #     #     'tiger-alt-start', 'tmaze_5_two_thirds_up', 'example_7', '4x3.95',
    #     #     'cheese.95', 'network', 'shuttle.95', 'paint.95'
    #     #     # 'hallway'
    #     #     # 'bridge-repair',
    #     # ],
    #     'policy_optim_alg': 'policy_grad',
    #     'value_type': 'q',
    #     'error_type': 'l2',
    #     'alpha': 1.,
    #     'mi_steps': 20000,
    #     'pi_steps': 10000,
    #     'optimizer': 'adam',
    #     'lr': 0.01,
    #     'n_mem_states': 8,
    #     'mi_iterations': 1,
    #     'random_policies': 10,
    #     'n_seeds': 1,
    #     'seed': [2024 + i for i in range(10)],
    #     'platform': 'gpu'
    # }

    ]
}
