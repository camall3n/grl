"""
This file runs a memory iteration with a batch of randomized initial policies,
as well as the TD optimal policy, on a list of different measures.

"""
import argparse
import logging
import pathlib
from time import time

import numpy as np
import jax
from jax.config import config

from grl.environment import load_pomdp

if __name__ == "__main__":
    start_time = time()
    # jax.disable_jit(True)

    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='example_11', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--mi_iterations', type=int, default=1,
                        help='For memory iteration, how many iterations of memory iterations do we do?')
    parser.add_argument('--mi_steps', type=int, default=50000,
                        help='For memory iteration, how many steps of memory improvement do we do per iteration?')
    parser.add_argument('--pi_steps', type=int, default=50000,
                        help='For memory iteration, how many steps of policy improvement do we do per iteration?')


    parser.add_argument('--policy_optim_alg', type=str, default='policy_iter',
                        help='policy improvement algorithm to use. "policy_iter" - policy iteration, "policy_grad" - policy gradient, '
                             '"discrep_max" - discrepancy maximization, "discrep_min" - discrepancy minimization')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')

    parser.add_argument('--random_policies', default=64, type=int,
                        help='How many random policies do we learn memory over?')
    parser.add_argument('--n_mem_states', default=2, type=int,
                        help='for memory_id = 0, how many memory states do we have?')

    parser.add_argument('--lambda_0', default=0., type=float,
                        help='First lambda parameter for lambda-discrep')
    parser.add_argument('--lambda_1', default=1., type=float,
                        help='Second lambda parameter for lambda-discrep')

    parser.add_argument('--alpha', default=1., type=float,
                        help='Temperature parameter, for how uniform our lambda-discrep weighting is')
    parser.add_argument('--objectives', default=['discrep', 'tde', 'tde_residual'])
