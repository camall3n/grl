import argparse
import logging
import pathlib
import time

import numpy as np
import jax
import matplotlib.pyplot as plt
import seaborn as sns

from grl.environment import *
from grl.mdp import MDP, AbstractMDP
from grl.td_lambda import td_lambda
from grl.policy_eval import PolicyEval
from grl.memory import memory_cross_product
from grl.grad import do_grad
from grl.utils import pformat_vals, RTOL

np.set_printoptions(precision=4, suppress=True)

def run_algos(spec, method, n_random_policies, use_grad, n_episodes):
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    policies = spec['Pi_phi']
    if 'T_mem' in spec.keys():
        amdp = memory_cross_product(amdp, spec['T_mem'])
        policies = spec['Pi_phi_x']
    if n_random_policies > 0:
        policies = amdp.generate_random_policies(n_random_policies)
    pe = PolicyEval(amdp)
    discrepancy_ids = []

    for i, pi in enumerate(policies):
        logging.info(f'\n\n\n======== policy id: {i} ========')
        logging.info(f'\npi:\n {pi}')
        pi_ground = amdp.get_ground_policy(pi)
        logging.info(f'\npi_ground:\n {pi_ground}')

        if method == 'a' or method == 'b':
            logging.info('\n--- Analytical ---')
            mdp_vals_a, mc_vals_a, td_vals_a = pe.run(pi)
            logging.info(f'\nmdp:\n {pformat_vals(mdp_vals_a)}')
            logging.info(f'mc*:\n {pformat_vals(mc_vals_a)}')
            logging.info(f'td:\n {pformat_vals(td_vals_a)}')
            discrep = {
                'v': np.abs(td_vals_a['v'] - mc_vals_a['v']),
                'q': np.abs(td_vals_a['q'] - mc_vals_a['q']),
            }
            logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

            # If using memory, for mc and td, also aggregate obs-mem values into
            # obs values according to visitation ratios
            if 'T_mem' in spec.keys():
                occupancy_x = pe._get_occupancy()
                n_mem_states = spec['T_mem'].shape[-1]
                n_og_obs = amdp.n_obs // n_mem_states # number of obs in the original (non cross product) amdp

                # These operations are within the cross producted space
                ob_counts_x = amdp.phi.T @ occupancy_x
                ob_sums_x = ob_counts_x.reshape(n_og_obs, n_mem_states).sum(1)
                w_x = ob_counts_x / ob_sums_x.repeat(n_mem_states)

                logging.info('\n--- Cross product info')
                logging.info(f'ob-mem occupancy:\n {ob_counts_x}')
                logging.info(f'ob-mem weights:\n {w_x}')

                logging.info('\n--- Aggregation from obs-mem values (above) to obs values (below)')
                n_actions = mc_vals_a['q'].shape[0]
                mc_vals_x = {}
                td_vals_x = {}

                mc_vals_x['v'] = (mc_vals_a['v'] * w_x).reshape(n_og_obs, n_mem_states).sum(1)
                mc_vals_x['q'] = (mc_vals_a['q'] * w_x).reshape(n_actions, n_og_obs,
                                                                n_mem_states).sum(2)
                td_vals_x['v'] = (td_vals_a['v'] * w_x).reshape(n_og_obs, n_mem_states).sum(1)
                td_vals_x['q'] = (td_vals_a['q'] * w_x).reshape(n_actions, n_og_obs,
                                                                n_mem_states).sum(2)
                # logging.info(f'\nmdp:\n {pformat_vals(mdp_vals)}')
                logging.info(f'mc*:\n {pformat_vals(mc_vals_x)}')
                logging.info(f'td:\n {pformat_vals(td_vals_x)}')
                discrep = {
                    'v': np.abs(td_vals_x['v'] - mc_vals_x['v']),
                    'q': np.abs(td_vals_x['q'] - mc_vals_x['q']),
                }
                logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

            # Check if there are discrepancies in V or Q
            # V takes precedence
            value_type = None
            if not np.allclose(mc_vals_a['v'], td_vals_a['v'], rtol=RTOL):
                value_type = 'v'
            elif not np.allclose(mc_vals_a['q'], td_vals_a['q'], rtol=RTOL):
                value_type = 'q'

            if value_type:
                discrepancy_ids.append(i)
                if use_grad:
                    do_grad(spec, pi, grad_type=use_grad, value_type=value_type)

        if method == 's' or method == 'b':
            # Sampling
            logging.info('\n\n--- Sampling ---')
            # MDP
            v, q = td_lambda(
                mdp,
                pi_ground,
                lambda_=1,
                alpha=0.01,
                n_episodes=n_episodes,
            )
            mdp_vals_s = {
                'v': v,
                'q': q,
            }

            # TD(1)
            v, q = td_lambda(
                amdp,
                pi,
                lambda_=1,
                alpha=0.01,
                n_episodes=n_episodes,
            )
            mc_vals_s = {
                'v': v,
                'q': q,
            }

            # TD(0)
            v, q = td_lambda(
                amdp,
                pi,
                lambda_=0,
                alpha=0.01,
                n_episodes=n_episodes,
            )
            td_vals_s = {
                'v': v,
                'q': q,
            }

            logging.info(f'mdp:\n {pformat_vals(mdp_vals_s)}')
            logging.info(f'mc:\n {pformat_vals(mc_vals_s)}')
            logging.info(f'td:\n {pformat_vals(td_vals_s)}')
            discrep = {
                'v': np.abs(td_vals_s['v'] - mc_vals_s['v']),
                'q': np.abs(td_vals_s['q'] - mc_vals_s['q']),
            }
            logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

    logging.info('\nTD-MC* Discrepancy ids:')
    logging.info(f'{discrepancy_ids}')
    logging.info(f'({len(discrepancy_ids)}/{len(policies)})')

def heatmap(spec, discrep_type='l2', num_ticks=5):
    """
    (Currently have to adjust discrep_type and num_ticks above directly)
    """
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    policy_eval = PolicyEval(amdp, verbose=False)

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
                loss_fn = policy_eval.mse_loss_q
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
    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='example_11', type=str,
        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--method', default='a', type=str,
        help='"a"-analytical, "s"-sampling, "b"-both')
    parser.add_argument('--n_random_policies', default=0, type=int,
        help='number of random policies to eval; if set (>0), overrides Pi_phi')
    parser.add_argument('--use_memory', default=None, type=int,
        help='use memory function during policy eval if set')
    parser.add_argument('--use_grad', default=None, type=str,
        help='find policy ("p") or memory ("m") that minimizes any discrepancies by following gradient (currently using analytical discrepancy)')
    parser.add_argument('--heatmap', action='store_true',
        help='generate a policy-discrepancy heatmap for the given POMDP')
    parser.add_argument('--n_episodes', default=500, type=int,
        help='number of rollouts to run')
    parser.add_argument('--log', action='store_true',
        help='save output to logs/')

    parser.add_argument('--tmaze_corridor_length', default=5, type=int,
                        help='[T-MAZE] length of t-maze corridor')

    parser.add_argument('--seed', default=None, type=int,
        help='seed for random number generators')
    parser.add_argument('-f', '--fool-ipython') # hack to allow running in ipython notebooks
    # yapf:enable

    global args
    args = parser.parse_args()
    del args.fool_ipython

    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[logging.StreamHandler()])
    logging.getLogger().setLevel(logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        mem_part = 'no_memory'
        if args.use_memory is not None and args.use_memory > 0:
            mem_part = f'memory_{args.use_memory}'
        name = f'logs/{args.spec}-{mem_part}-{time.time()}.log'
        rootLogger.addHandler(logging.FileHandler(name))

    if args.seed:
        np.random.seed(args.seed)
        jax.random.PRNGKey(args.seed)

    # Get POMDP definition
    spec = environment.load_spec(args.spec, memory_id=args.use_memory, tmaze_corridor_length=args.tmaze_corridor_length)
    logging.info(f'spec:\n {args.spec}\n')
    logging.info(f'T:\n {spec["T"]}')
    logging.info(f'R:\n {spec["R"]}')
    logging.info(f'gamma: {spec["gamma"]}')
    logging.info(f'p0:\n {spec["p0"]}')
    logging.info(f'phi:\n {spec["phi"]}')
    logging.info(f'Pi_phi:\n {spec["Pi_phi"]}')
    if 'T_mem' in spec.keys():
        logging.info(f'T_mem:\n {spec["T_mem"]}')
    if 'Pi_phi_x' in spec.keys():
        logging.info(f'Pi_phi_x:\n {spec["Pi_phi_x"]}')

    logging.info(f'n_episodes:\n {args.n_episodes}')

    # Run
    if args.heatmap:
        heatmap(spec)
    else:
        run_algos(spec, args.method, args.n_random_policies, args.use_grad, args.n_episodes)
