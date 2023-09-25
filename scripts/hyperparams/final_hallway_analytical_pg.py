from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'args': [{
        'spec': 'hallway',
        'policy_optim_alg': 'policy_grad',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'objective': 'obs_space',
        # 'objective': ['discrep', 'obs_space'],
        # 'mi_steps': 20000,
        'mi_steps': 5000,
        # 'pi_steps': 10000,
        'pi_steps': 5000,
        'optimizer': 'adam',
        'lr': 0.01,
        'use_memory': 0,
        'n_mem_states': [2, 4, 8],
        'mi_iterations': 1,
        'platform': 'gpu',
        'seed': [2020 + i for i in range(30)],
    }]
}
