import argparse
import logging
import pathlib
from time import time

import numpy as np
import jax
from jax.config import config

from grl.environment import load_spec
from grl.environment.policy_lib import get_start_pi
from grl.utils.file_system import results_path, numpyify_and_save
from grl.memory_iteration import run_memory_iteration

def add_tmaze_hyperparams(parser: argparse.ArgumentParser):
    # hyperparams for tmaze_hperparams
    parser.add_argument('--tmaze_corridor_length',
                        default=None,
                        type=int,
                        help='Length of corridor for tmaze_hyperparams')
    parser.add_argument('--tmaze_discount',
                        default=None,
                        type=float,
                        help='Discount rate for tmaze_hyperparams')
    parser.add_argument('--tmaze_junction_up_pi',
                        default=None,
                        type=float,
                        help='probability of traversing up at junction for tmaze_hyperparams')
    return parser

if __name__ == '__main__':
    start_time = time()

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
    parser.add_argument('--init_pi', default=None, type=str,
                        help='Do we initialize our policy to something?')
    parser.add_argument('--use_memory', default=None, type=str,
        help='use memory function during policy eval if set')
    parser.add_argument('--mem_fuzz', default=0.1, type=float,
                        help='For the fuzzy identity memory function, how much fuzz do we add?')
    parser.add_argument('--n_mem_states', default=2, type=int,
                        help='for memory_id = 0, how many memory states do we have?')
    parser.add_argument('--alpha', default=1., type=float,
                        help='Temperature parameter, for how uniform our lambda-discrep weighting is')
    parser.add_argument('--flip_count_prob', action='store_true',
                        help='Do we "invert" our count probabilities for our memory loss?')
    parser.add_argument('--value_type', default='q', type=str,
                        help='Do we use (v | q) for our discrepancies?')
    parser.add_argument('--error_type', default='l2', type=str,
                        help='Do we use (l2 | abs) for our discrepancies?')
    parser.add_argument('--objective', default='discrep', type=str,
                        help='What objective are we trying to optimize? (discrep | magnitude)')
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='(POLICY ITERATION AND TMAZE_EPS_HYPERPARAMS ONLY) What epsilon do we use?')
    parser.add_argument('--log', action='store_true',
        help='save output to logs/')
    parser.add_argument('--experiment_name', default=None, type=str,
        help='name of the experiment. Results saved to results/{experiment_name} directory if not None. Else, save to results directory directly.')
    parser.add_argument('--platform', default='cpu', type=str,
                        help='What platform do we run things on? (cpu | gpu)')
    parser.add_argument('--seed', default=None, type=int,
        help='seed for random number generators')
    parser = add_tmaze_hyperparams(parser)
    # yapf:enable

    global args
    args = parser.parse_args()

    # configs
    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        mem_part = 'no_memory'
        if args.use_memory is not None and args.use_memory.isdigit() and int(args.use_memory) > 0:
            mem_part = f'memory_{args.use_memory}'
        name = f'logs/{args.spec}-{mem_part}-{time()}.log'
        rootLogger.addHandler(logging.FileHandler(name))

    rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Run
    # Get POMDP definition
    spec = load_spec(args.spec,
                     memory_id=args.use_memory,
                     n_mem_states=args.n_mem_states,
                     corridor_length=args.tmaze_corridor_length,
                     discount=args.tmaze_discount,
                     junction_up_pi=args.tmaze_junction_up_pi,
                     epsilon=args.epsilon,
                     fuzz=args.mem_fuzz)

    logging.info(f'spec:\n {args.spec}\n')
    logging.info(f'T:\n {spec["T"]}')
    logging.info(f'R:\n {spec["R"]}')
    logging.info(f'gamma: {spec["gamma"]}')
    logging.info(f'p0:\n {spec["p0"]}')
    logging.info(f'phi:\n {spec["phi"]}')

    if 'mem_params' in spec.keys():
        logging.info(f'mem_params:\n {spec["mem_params"]}')
    if 'Pi_phi_x' in spec.keys():
        logging.info(f'Pi_phi_x:\n {spec["Pi_phi_x"]}')
    if 'Pi_phi' in spec and spec['Pi_phi'] is not None:
        logging.info(f'Pi_phi:\n {spec["Pi_phi"]}')

    results_path = results_path(args)

    pi_params = None
    if args.init_pi is not None:
        pi_params = get_start_pi(args.init_pi, spec=spec)
    logs, agent = run_memory_iteration(spec,
                                       pi_lr=args.lr,
                                       mi_lr=args.lr,
                                       rand_key=rand_key,
                                       mi_iterations=args.mi_iterations,
                                       policy_optim_alg=args.policy_optim_alg,
                                       mi_steps=args.mi_steps,
                                       pi_steps=args.pi_steps,
                                       value_type=args.value_type,
                                       error_type=args.error_type,
                                       objective=args.objective,
                                       alpha=args.alpha,
                                       epsilon=args.epsilon,
                                       pi_params=pi_params,
                                       flip_count_prob=args.flip_count_prob)

    info = {'logs': logs, 'args': args.__dict__}
    agents_dir = results_path.parent / 'agents'
    agents_dir.mkdir(exist_ok=True)

    agents_path = agents_dir / f'{results_path.stem}.pkl'
    np.save(agents_path, agent)

    end_time = time()
    run_stats = {'start_time': start_time, 'end_time': end_time}
    info['run_stats'] = run_stats

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
