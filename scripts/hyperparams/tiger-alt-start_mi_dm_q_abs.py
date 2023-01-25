hparams = {
    'file_name':
        'runs_tiger-alt-start_mi_dm_q_abs.txt',
    'args': [{
        'algo': 'mi',
        'spec': 'tiger-alt-start',
        'policy_optim_alg': 'dm',
        'value_type': 'q',
        'error_type': 'abs',
        'method': 'a',
        'lr': 1,
        'use_memory': 0,
        'use_grad': 'm',
        'mi_steps': 200000,
        'pi_steps': 200000,
        # 'n_mem_states': [2, 4, 6],
        'n_mem_states': 2,
        'mi_iterations': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
