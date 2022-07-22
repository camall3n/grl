import logging

from .mdp import MDP, AbstractMDP
from .policy_eval import PolicyEval
from .memory import memory_cross_product
from .utils import pformat_vals

import numpy as np
from jax import grad

def do_grad(spec, pi_abs, grad_type, no_gamma, value_type='v', discrep_type='l2', lr=1):
    """
    :param spec:         spec
    :param pi_abs:       pi_abs
    :param lr:           learning rate
    :param grad_type:    'p'olicy or 'm'emory
    :param no_gamma:     no gamma in occupancy
    :param value_type:   'v' or 'q'
    :param discrep_type: 'l2' or 'max'
        - 'l2' uses MSE over all obs(/actions)
        - 'max' uses the highest individual absolute difference across obs(/actions) 
        - (see policy_eval.py)
        - Currently has to be adjusted above directly
    """

    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])
    policy_eval = PolicyEval(amdp, no_gamma)

    if grad_type == 'p':
        params = pi_abs
        if 'T_mem' in spec.keys():
            amdp = memory_cross_product(amdp, spec['T_mem'])
            policy_eval = PolicyEval(amdp, no_gamma)

        if discrep_type == 'l2':
            loss_fn = policy_eval.mse_loss
        elif discrep_type == 'max':
            loss_fn = policy_eval.max_loss

    elif grad_type == 'm':
        if 'T_mem' not in spec.keys():
            raise ValueError(
                'Must include memory with "--use_memory <id>" to do gradient with memory')
        params = spec['T_mem']
        loss_fn = policy_eval.memory_loss

    policy_eval.verbose = False
    logging.info(f'\nStarting discrep:\n {loss_fn(params, value_type, pi_abs=pi_abs)}')

    i = 0
    done_count = 0
    old_params = params

    while done_count < 5:
        i += 1

        params_grad = grad(loss_fn, argnums=0)(params, value_type, pi_abs=pi_abs)
        old_params = params
        params -= lr * params_grad

        # Normalize (assuming params are probability distribution)
        params = params.clip(0, 1)
        denom = params.sum(axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1, denom) # Avoid divide by zero (there may be a better way)
        params /= denom

        if i % 10 == 0:
            print('\n\n')
            print('Gradient iteration', i)
            print('params_grad\n', params_grad)
            print()
            print('params\n', params)

        if np.allclose(old_params, params):
            done_count += 1
        else:
            done_count = 0

    # Log results
    logging.info(f'\n\n---- GRAD RESULTS ----\n')
    logging.info(f'Final gradient params:\n {params}')
    logging.info(f'in {i} gradient steps with lr={lr}')

    old_amdp = policy_eval.amdp
    if grad_type == 'm':
        policy_eval.amdp = memory_cross_product(amdp, params)
    policy_eval.verbose = True
    mdp_vals, amdp_vals, td_vals = policy_eval.run(pi_abs)
    logging.info(f'\nFinal vals using grad_type {grad_type} on value_type {value_type}')
    logging.info(f'mdp:\n {pformat_vals(mdp_vals)}')
    logging.info(f'mc*:\n {pformat_vals(amdp_vals)}')
    logging.info(f'td:\n {pformat_vals(td_vals)}')
    policy_eval.amdp = old_amdp
    logging.info(f'discrep:\n {loss_fn(params, value_type, pi_abs=pi_abs)}')

    return params
