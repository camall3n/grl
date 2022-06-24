import logging

from .utils import pformat_vals

import numpy as np
from jax import grad

def do_grad(policy_eval, pi_abs, value_type='v', discrep_type='l2', lr=1):
    """
    :param policy_eval:   PolicyEval object
    :param pi_abs:        policy over abstract state space
    :param lr:            learning rate
    :param value_type:  'v' or 'q'
    :param discrep_type:   'l2' or 'max'
        - 'l2' uses MSE over all obs(/actions)
        - 'max' uses the highest individual absolute difference across obs(/actions) 
        - (see policy_eval.py)
        - Currently has to be adjusted above directly
    """

    if value_type == 'v':
        if discrep_type == 'l2':
            grad_fn = grad(policy_eval.mse_loss_v)
        elif discrep_type == 'max':
            grad_fn = grad(policy_eval.max_loss_v)
    elif value_type == 'q':
        if discrep_type == 'l2':
            grad_fn = grad(policy_eval.mse_loss_q)
        elif discrep_type == 'max':
            grad_fn = grad(policy_eval.max_loss_q)

    policy_eval.verbose = False
    old_pi = pi_abs
    i = 0
    done_count = 0

    while done_count < 5:
        i += 1
        if i % 10 == 0:
            print('Gradient iteration', i)

        pi_grad = grad_fn(pi_abs)
        old_pi = pi_abs
        pi_abs -= lr * pi_grad

        # Normalize
        pi_abs = pi_abs.clip(0, 1)
        denom = pi_abs.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom) # Avoid divide by zero (there may be a better way)
        pi_abs /= denom

        if np.allclose(old_pi, pi_abs):
            done_count += 1
        else:
            done_count = 0

    logging.info(f'Final gradient pi:\n {pi_abs}')
    logging.info(f'in {i} gradient steps with lr={lr}')

    # policy_eval.verbose = True
    mdp_vals, amdp_vals, td_vals = policy_eval.run(pi_abs)
    logging.info('\nFinal vals using gradient pi')
    logging.info(f'mdp:\n {pformat_vals(mdp_vals)}')
    logging.info(f'mc*:\n {pformat_vals(amdp_vals)}')
    logging.info(f'td:\n {pformat_vals(td_vals)}')

    return pi_abs
