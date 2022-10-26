import logging
from jax.nn import softmax

from .mdp import MDP, AbstractMDP
from .policy_eval import PolicyEval
from .memory import memory_cross_product
from .utils import pformat_vals

import numpy as np

def do_grad(spec, pi_abs, grad_type, value_type='v', discrep_type='l2', lr=1):
    """
    :param spec:         spec
    :param pi_abs:       pi_abs
    :param lr:           learning rate
    :param grad_type:    'p'olicy or 'm'emory
    :param value_type:   'v' or 'q'
    :param discrep_type: 'l2' or 'max'
        - 'l2' uses MSE over all obs(/actions)
        - 'max' uses the highest individual absolute difference across obs(/actions) 
        - (see policy_eval.py)
        - Currently has to be adjusted above directly
    """

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    policy_eval = PolicyEval(amdp, discrep_type=discrep_type)

    if grad_type == 'p':
        params = pi_abs
        if 'mem_params' in spec.keys():
            amdp = memory_cross_product(amdp, spec['mem_params'])
            policy_eval = PolicyEval(amdp, discrep_type=discrep_type)

        update = policy_eval.policy_update
        if discrep_type == 'l2':
            loss_fn = policy_eval.mse_loss
        elif discrep_type == 'max':
            loss_fn = policy_eval.max_loss
        else:
            raise NotImplementedError

    elif grad_type == 'm':
        if 'mem_params' not in spec.keys():
            raise ValueError(
                'Must include memory with "--use_memory <id>" to do gradient with memory')
        params = spec['mem_params']
        update = policy_eval.memory_update
        loss_fn = policy_eval.memory_loss
    else:
        raise NotImplementedError

    policy_eval.verbose = False
    logging.info(f'\nStarting discrep:\n {loss_fn(params, value_type, pi_abs=pi_abs)}')

    i = 0
    done_count = 0
    # old_params = params

    while done_count < 5:
        i += 1

        old_params = params
        loss, new_params = update(params, value_type, lr, pi_abs)
        params = new_params

        if i % 100 == 0:
            # print('\n\n')
            print(f'Gradient iteration {i}, loss: {loss.item():.4f}')
            # print('params_grad\n', params_grad)
            # print()
            # print('params\n', params)

        if np.allclose(old_params, params, atol=1e-10):
            done_count += 1
        else:
            done_count = 0

    # Log results
    logging.info(f'\n\n---- GRAD RESULTS ----\n')
    logging.info(f'-Final gradient params:\n {softmax(params, axis=-1)}')
    logging.info(f'in {i} gradient steps with lr={lr}')

    old_amdp = policy_eval.amdp
    if grad_type == 'm':
        policy_eval.amdp = memory_cross_product(amdp, params)
    policy_eval.verbose = True
    mdp_vals, amdp_vals, td_vals = policy_eval.run(pi_abs)
    logging.info(f'\n-Final vals using grad_type {grad_type} on value_type {value_type}')
    logging.info(f'mdp:\n {pformat_vals(mdp_vals)}')
    logging.info(f'mc*:\n {pformat_vals(amdp_vals)}')
    logging.info(f'td:\n {pformat_vals(td_vals)}')
    policy_eval.amdp = old_amdp
    # logging.info(f'discrep:\n {loss_fn(params, value_type, pi_abs=pi_abs)}')

    return params
