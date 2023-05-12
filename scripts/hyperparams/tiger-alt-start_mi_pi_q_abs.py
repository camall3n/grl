hparams = {
    'file_name':
        'runs_tiger-alt-start_mi_pi_q_abs.txt',
    'args': [{
        'algo': 'mi',
        'spec': 'tiger-alt-start',
        'policy_optim_alg': 'pi',
        'value_type': 'q',
        'error_type': 'abs',
        'method': 'a',
        'lr': 1,
        'use_memory': 0,
        'use_grad': 'm',
<<<<<<< HEAD
        # 'n_mem_states': [2, 4, 6],
        'n_mem_states': 2,
=======
        'n_mem_states': [4, 6],
        # 'n_mem_states': 2,
>>>>>>> abs_upper_bound
        'mi_iterations': 2,
        'seed': [2020 + i for i in range(10)],
    }]
}
