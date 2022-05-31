import argparse
import logging
import pathlib
import time

import numpy as np

from .environment import *
from .mdp import MDP, AbstractMDP
from .mc import mc
from .policy_eval import PolicyEval

def run_algos(spec, no_gamma, n_steps, max_rollout_steps):
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])

    # Policy Eval
    logging.info('\n===== Policy Eval =====')
    for pi in spec['Pi_phi']:
        pe = PolicyEval(amdp, pi)
        mdp_vals, amdp_vals, td_vals = pe.run(no_gamma)
        logging.info(f'\npi: {pi}')
        logging.info(f'\nmdp: {mdp_vals}')
        logging.info(f'amdp: {amdp_vals}')
        logging.info(f'td: {td_vals}')
        logging.info('\n-----------')

    # Sampling
    # logging.info('\n\n===== Sampling =====')
    # for pi in spec['Pi_phi']:
    #     logging.info(f'\npi: {pi}')

    #     # MC*
    #     # MDP
    #     v, q, pi = mc(mdp, pi, p0=spec['p0'], alpha=0.01, epsilon=0, mc_states='all', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
    #     logging.info("\n- mc_states: all")
    #     logging.info(f'mdp: {v}')

    #     # AMDP
    #     v, q, pi = mc(amdp, pi, p0=spec['p0'], alpha=0.001, epsilon=0, mc_states='all', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
    #     logging.info(f'amdp: {v}')

    #     # MC1
    #     # ADMP
    #     logging.info("\n- mc_states: first")
    #     v, q, pi = mc(amdp, pi, p0=spec['p0'], alpha=0.01, epsilon=0, mc_states='first', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
    #     logging.info(f'amdp: {v}')

    #     logging.info('\n-----------')

if __name__ == '__main__':
    # Usage: python -m grl.run --spec example_11 --log

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='example_11', type=str)
    parser.add_argument('--no_gamma', action='store_true',
                        help='do not discount the weighted average value expectation in policy eval')
    parser.add_argument('--n_steps', default=20000, type=int,
                        help='number of rollouts to run')
    parser.add_argument('--max_rollout_steps', default=None, type=int,
                        help='max steps for mc rollouts')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('-f', '--fool-ipython')# hack to allow running in ipython notebooks
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    del args.fool_ipython

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        rootLogger.addHandler(logging.FileHandler(f'logs/{args.spec}-{time.time()}.log'))

    if args.seed:
        np.random.seed(args.seed)

    # Get POMDP definition
    spec = environment.load_spec(args.spec)
    logging.info(f'spec:\n {args.spec}\n')
    logging.info(f'T:\n {spec["T"]}')
    logging.info(f'R:\n {spec["R"]}')
    logging.info(f'gamma: {spec["gamma"]}')
    logging.info(f'p0:\n {spec["p0"]}')
    logging.info(f'phi:\n {spec["phi"]}')
    logging.info(f'Pi_phi:\n {spec["Pi_phi"]}')
    logging.info(f'n_steps:\n {args.n_steps}')
    logging.info(f'max_rollout_steps:\n {args.max_rollout_steps}')

    # Run algos
    run_algos(spec, args.no_gamma, args.n_steps, args.max_rollout_steps)
