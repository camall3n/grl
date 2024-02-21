from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m grl.batch_run_interleave',
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
        'alpha': [0., 1.],
        # 'alpha': 0.,
        'objective': 'discrep',
        'mi_steps': 20000,
        'optimizer': 'adam',
        'lr': 0.01,
        'use_memory': 0,
        'n_mem_states': [2, 4, 8],
        'mi_iterations': 1,
        'platform': 'gpu',
        'n_seeds': 10,
    }]
}
