import logging

from .utils import pformat_vals

import numpy as np
from jax import grad

def do_grad(policy_eval, pi_abs, no_gamma, lr=1):
    """
    :param policy_eval: PolicyEval object
    :param pi_abs:      policy over abstract state space
    :param no_gamma:    passed to policy_eval run() func
    :param lr:          learning rate
    """
    def mse_loss(pi):
        _, amdp_vals, td_vals = policy_eval.run(pi, no_gamma)
        diff = amdp_vals['v'] - td_vals['v'] # TODO: q vals?
        return (diff**2).mean()

    policy_eval.verbose = False
    old_pi = pi_abs
    i = 0
    done_count = 0
    while done_count < 5:
        i += 1
        if i % 10 == 0:
            print('Gradient iteration', i)

        pi_grad = grad(mse_loss)(pi_abs)
        old_pi = pi_abs
        pi_abs -= lr * pi_grad

        # Normalize
        pi_abs = pi_abs.clip(0,1)
        pi_abs /= pi_abs.sum(axis=1, keepdims=True)

        if np.allclose(old_pi, pi_abs):
            done_count += 1
        else:
            done_count = 0

    logging.info(f'Final gradient pi:\n {pi_abs}')
    logging.info(f'in {i} gradient steps with lr={lr}')

    # policy_eval.verbose = True
    mdp_vals, amdp_vals, td_vals = policy_eval.run(pi_abs, no_gamma)
    logging.info('\nFinal vals using gradient pi')
    logging.info(f'mdp:\n {pformat_vals(mdp_vals)}')
    logging.info(f'mc*:\n {pformat_vals(amdp_vals)}')
    logging.info(f'td:\n {pformat_vals(td_vals)}')

    return pi_abs
