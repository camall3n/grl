import argparse
import logging
import pathlib
import time

import numpy as np
import jax
import matplotlib.pyplot as plt
import seaborn as sns

from .environment import *
from .mdp import MDP, AbstractMDP
from .mc import mc
from .policy_eval import PolicyEval
from .grad import do_grad
from .utils import pformat_vals, RTOL

def run_algos(spec, no_gamma, n_random_policies, use_grad, n_steps, max_rollout_steps):
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])

    # Policy Eval
    logging.info('\n===== Policy Eval =====')
    policies = spec['Pi_phi']
    if n_random_policies > 0:
        policies = amdp.generate_random_policies(n_random_policies)

    pe = PolicyEval(amdp, no_gamma)
    discrepancy_ids = []
    for i, pi in enumerate(policies):
        logging.info(f'\nid: {i}')
        logging.info(f'\npi:\n {pi}')
        mdp_vals, amdp_vals, td_vals = pe.run(pi)
        logging.info(f'\nmdp:\n {pformat_vals(mdp_vals)}')
        logging.info(f'mc*:\n {pformat_vals(amdp_vals)}')
        logging.info(f'td:\n {pformat_vals(td_vals)}')

        # Check if there are discrepancies in V or Q
        # V takes precedence
        value_type = None
        if not np.allclose(amdp_vals['v'], td_vals['v'], rtol=RTOL):
            value_type = 'v'
        elif not np.allclose(amdp_vals['q'], td_vals['q'], rtol=RTOL):
            value_type = 'q'

        if value_type:
            discrepancy_ids.append(i)
            if use_grad:
                do_grad(pe, pi, value_type=value_type)

        logging.info('\n-----------')

    logging.info('\nTD-MC* Discrepancy ids:')
    logging.info(f'{discrepancy_ids}')
    logging.info(f'({len(discrepancy_ids)}/{len(policies)})')

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

def heatmap(spec, discrep_type='l2', no_gamma=True, num_ticks=5):
    """
    (Currently have to adjust discrep_type and num_ticks above directly)
    """
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])
    policy_eval = PolicyEval(amdp, no_gamma, verbose=False)

    # Run for both v and q
    value_types = ['v', 'q']
    for value_type in value_types:
        if value_type == 'v':
            if discrep_type == 'l2':
                loss_fn = policy_eval.mse_loss_v
            elif discrep_type == 'max':
                loss_fn = policy_eval.max_loss_v
        elif value_type == 'q':
            if discrep_type == 'l2':
                loss_fn = policy_eval.max_loss_q
            elif discrep_type == 'max':
                loss_fn = policy_eval.max_loss_q

        discrepancies = []
        x = y = np.linspace(0, 1, num_ticks)
        for i in range(num_ticks):
            p = x[i]
            for j in range(num_ticks):
                q = y[j]
                pi = np.array([[p, 1 - p], [q, 1 - q], [0, 0]])
                discrepancies.append(loss_fn(pi))

                if (num_ticks * i + j + 1) % 10 == 0:
                    print(f'Calculating policy {num_ticks * i + j + 1}/{num_ticks * num_ticks}')

        ax = sns.heatmap(np.array(discrepancies).reshape((num_ticks, num_ticks)),
                         xticklabels=x.round(3),
                         yticklabels=y.round(3),
                         cmap='viridis')
        ax.invert_yaxis()
        ax.set(xlabel='2nd obs', ylabel='1st obs')
        ax.set_title(f'{args.spec}, {value_type}_values, {discrep_type}_loss')
        plt.show()

if __name__ == '__main__':
    # Usage: python -m grl.run --spec example_3 --no_gamma --log

    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='example_11', type=str,
        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--no_gamma', action='store_true',
        help='do not discount the occupancy expectation in policy eval')
    parser.add_argument('--n_random_policies', default=0, type=int,
        help='number of random policies to eval; if set (>0), overrides Pi_phi')
    parser.add_argument('--use_grad', action='store_true',
        help='find policy that minimizes any discrepancies by following gradient')
    parser.add_argument('--heatmap', action='store_true',
        help='generate a policy-discrepancy heatmap for the given POMDP')
    parser.add_argument('--n_steps', default=20000, type=int,
        help='number of rollouts to run')
    parser.add_argument('--max_rollout_steps', default=None, type=int,
        help='max steps for mc rollouts')
    parser.add_argument('--log', action='store_true',
        help='save output to logs/')
    parser.add_argument('--seed', default=None, type=int,
        help='seed for random number generators')
    parser.add_argument('-f', '--fool-ipython') # hack to allow running in ipython notebooks
    # yapf:enable

    global args
    args = parser.parse_args()
    del args.fool_ipython

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        rootLogger.addHandler(logging.FileHandler(f'logs/{args.spec}-{time.time()}.log'))

    if args.seed:
        np.random.seed(args.seed)
        jax.random.PRNGKey(args.seed)

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

    # Run
    if args.heatmap:
        heatmap(spec, no_gamma=args.no_gamma)
    else:
        run_algos(spec, args.no_gamma, args.n_random_policies, args.use_grad, args.n_steps,
                  args.max_rollout_steps)
