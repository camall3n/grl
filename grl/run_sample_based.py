import argparse

import numpy as np
import jax
from jax.config import config

from grl.environment import load_spec
from grl.mdp import AbstractMDP, MDP
from grl.utils.file_system import results_path, numpyify_and_save

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='example_11', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--algo', default='dqn', type=str,
                        help='Baseline algorithm to evaluate')
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='(POLICY ITERATION AND TMAZE_EPS_HYPERPARAMS ONLY) What epsilon do we use?')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')

    args = parser.parse_args()

    # configs
    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Run
    # Get POMDP definition
    spec = load_spec(args.spec,
                     corridor_length=args.tmaze_corridor_length,
                     discount=args.tmaze_discount,
                     junction_up_pi=args.tmaze_junction_up_pi,
                     epsilon=args.epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])

    results_path = results_path(args)
    agents_dir = results_path.parent / 'agent'
    agents_dir.mkdir(exist_ok=True)

    agents_path = agents_dir / f'{results_path.stem}'
