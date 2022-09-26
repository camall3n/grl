import argparse
import logging
import pathlib
import time
from os import listdir
import os.path

import numpy as np
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .environment import load_spec
from .environment.pomdp_file import POMDPFile
from .mdp import MDP, AbstractMDP
from .td_lambda import td_lambda
from .policy_eval import PolicyEval
from .memory import memory_cross_product, generate_1bit_mem_fns, generate_mem_fn
from .grad import do_grad
from .utils import pformat_vals, RTOL

np.set_printoptions(precision=4, suppress=True)

def run_algos(spec, method='a', n_random_policies=0, use_grad=False, n_episodes=500):
    """
    Runs MDP, POMDP TD, and POMDP MC evaluations on given spec using given method.
    See args in __main__ function for param details.
    """
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    policies = spec['Pi_phi']
    if 'T_mem' in spec.keys():
        amdp = memory_cross_product(amdp, spec['T_mem'])
        policies = spec['Pi_phi_x']
    if n_random_policies > 0:
        policies = amdp.generate_random_policies(n_random_policies)
    pe = PolicyEval(amdp)
    discrepancies = [] # the discrepancy dict for each policy
    discrepancy_ids = [] # Pi_phi indices (policies) where there was a discrepancy

    for i, pi in enumerate(policies):
        logging.info(f'\n\n\n======== policy id: {i} ========')
        logging.info(f'\npi:\n {pi}')
        if 'T_mem' in spec.keys():
            logging.info(f'\nT_mem:\n {spec["T_mem"]}')
        pi_ground = amdp.get_ground_policy(pi)
        logging.info(f'\npi_ground:\n {pi_ground}')

        if method == 'a' or method == 'b':
            logging.info('\n--- Analytical ---')
            mdp_vals_a, mc_vals_a, td_vals_a = pe.run(pi)
            occupancy = pe._get_occupancy()
            pr_oa = (occupancy @ amdp.phi * pi.T)
            logging.info(f'\nmdp:\n {pformat_vals(mdp_vals_a)}')
            logging.info(f'mc*:\n {pformat_vals(mc_vals_a)}')
            logging.info(f'td:\n {pformat_vals(td_vals_a)}')
            discrep = {
                'v': np.abs(td_vals_a['v'] - mc_vals_a['v']),
                'q': np.abs(td_vals_a['q'] - mc_vals_a['q']),
            }
            discrep['q_sum'] = (discrep['q'] * pr_oa).sum()

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
                occ_obs = (occupancy_x @ amdp.phi).reshape(n_og_obs, n_mem_states).sum(-1)
                pi_obs = (pi.T * w_x).reshape(n_actions, n_og_obs, n_mem_states).sum(-1).T
                pr_oa = (occ_obs * pi_obs.T)
                discrep['q_sum'] = (discrep['q'] * pr_oa).sum()

                logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

            discrepancies.append(discrep)

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

    return discrepancies

def run_generated(dir, pomdp_id=None, mem_fn_id=None):
    # There needs to be at least one memory function that decreases the discrepancy
    # under each policy.
    # So we will track for each file, for each policy, whether a memory function has been found.

    if pomdp_id is None:
        # Runs algos on all pomdps defined in 'dir' using all 1 bit memory functions.

        # The objective is to determine whether there are any specs for which no 1 bit memory function
        # decreases an existing discrepancy.
        files = [f for f in listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        results = []
        for f in reversed(files):
            results.append(run_on_file(f'{dir}/{f}'))
    else:
        results = run_on_file(f'{dir}/{pomdp_id}.POMDP', mem_fn_id)
    return results

def run_on_file(filepath, mem_fn_id=None):
    spec = POMDPFile(f'{filepath}').get_spec()
    filename = os.path.basename(filepath)
    mdp_name = os.path.splitext(filename)[0]
    tag = os.path.split(os.path.dirname(filepath))[-1]

    logging.info(f'\n\n==========================================================')
    logging.info(f'GENERATED FILE: {mdp_name}')
    logging.info(f'==========================================================')

    # Discrepancies without memory.
    # List with one discrepancy dict ('v' and 'q') per policy.
    discrepancies_no_mem = run_algos(spec)

    path = f'grl/results/1bit_mem_conjecture_traj_weighted/{tag}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if mem_fn_id is None:
        results = []
        for mem_fn_id, T_mem in enumerate(
                tqdm(
                    generate_1bit_mem_fns(n_obs=spec['phi'].shape[-1],
                                          n_actions=spec['T'].shape[0]))):

            result = record_discrepancy_improvements(path, mdp_name, spec, mem_fn_id, T_mem,
                                                     discrepancies_no_mem)
            results.append(result)
    else:
        T_mem = generate_mem_fn(mem_fn_id,
                                n_mem_states=2,
                                n_obs=spec['phi'].shape[-1],
                                n_actions=spec['T'].shape[0])
        results = record_discrepancy_improvements(path, mdp_name, spec, mem_fn_id, T_mem,
                                                  discrepancies_no_mem)
    return results

def record_discrepancy_improvements(path, mdp_name, spec, mem_fn_id, T_mem, discrepancies_no_mem):
    """Create a file if the memory function improves the discrepancy"""
    spec['T_mem'] = T_mem # add memory
    spec['Pi_phi_x'] = [pi.repeat(2, axis=0)
                        for pi in spec['Pi_phi']] # expand policies to obs-mem space
    discrepancies_mem = run_algos(spec)

    # Check if this memory made the discrepancy decrease for each policy.
    # The mem and no_mem lists are in the same order of policies.
    n_policies = len(discrepancies_mem)
    mem_fn_improved_discrep = [False] * n_policies
    for policy_id in range(n_policies):
        disc_no_mem = discrepancies_no_mem[policy_id]
        disc_mem = discrepancies_mem[policy_id]

        def is_pareto_q_discrepancy_improvement(disc_mem, disc_no_mem) -> bool:
            something_improved = (~np.isclose(disc_mem['q'], disc_no_mem['q'])
                                  & (disc_mem['q'] < disc_no_mem['q'])).any()
            something_got_worse = (~np.isclose(disc_mem['q'], disc_no_mem['q'])
                                   & (disc_mem['q'] > disc_no_mem['q'])).any()
            if (something_improved and not something_got_worse):
                return True
            return False

        def is_traj_weighted_q_discrepancy_improvement(disc_mem, disc_no_mem) -> bool:
            improvement = (~np.isclose(disc_mem['q_sum'], disc_no_mem['q_sum'])
                           & (disc_mem['q_sum'] < disc_no_mem['q_sum']))
            if improvement:
                return True
            return False

        # if is_pareto_q_discrepancy_improvement(disc_mem, disc_no_mem):
        if is_traj_weighted_q_discrepancy_improvement(disc_mem, disc_no_mem):
            # Create file if discrepancy was reduced
            pathlib.Path(f'{path}/{mdp_name}_{policy_id}_{mem_fn_id}.txt').touch(exist_ok=True)
            mem_fn_improved_discrep[policy_id] = True
    return mem_fn_improved_discrep

def generate_pomdps(params):
    timestamp = str(time.time()).replace('.', '-')
    path = f'grl/environment/pomdp_files/generated/{timestamp}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for i in range(params['n_pomdps']):
        n_s = np.random.randint(params['min_n_s'], params['max_n_s'] + 1)
        n_o = np.random.randint(params['min_n_o'], params['max_n_o'] + 1)
        n_a = np.random.randint(params['min_n_a'], params['max_n_a'] + 1)
        gamma = np.random.random()
        amdp = AbstractMDP.generate(n_s, n_a, n_o, gamma=gamma)

        content = f'# Generation timestamp: {timestamp}\n'
        content += f'# with seed: {args.seed}\n'
        content += f'# with params: {params}\n\n'

        content += f'discount: {amdp.gamma}\n'
        content += 'values: reward\n'
        content += f'states: {amdp.n_states}\n'
        content += f'actions: {amdp.n_actions}\n'
        content += f'observations: {amdp.n_obs}\n'
        content += f'start: {str(amdp.p0)[1:-1]}\n\n' # remove array brackets

        # T
        for a in range(amdp.n_actions):
            content += f'T: {a}\n'
            for row in amdp.T[a]:
                content += f'{str(row)[1:-1]}\n' # remove array brackets

            content += '\n'

        # O
        content += 'O: *\n' # phi currently same for all actions
        for row in amdp.phi:
            content += f'{str(row)[1:-1]}\n' # remove array brackets

        content += '\n'

        # R
        for a in range(amdp.n_actions):
            for m, row in enumerate(amdp.R[a]):
                for n, val in enumerate(row):
                    content += f'R: {a} : {m} : {n} : * {val}\n'

            content += '\n'

        # Pi_phi
        policies = amdp.generate_random_policies(params['n_policies'])
        for pi in policies:
            content += f'Pi_phi:\n'
            for row in pi:
                content += f'{str(row)[1:-1]}\n' # remove array brackets

            content += '\n'

        with open(f'{path}/{i}.POMDP', 'w') as f:
            f.write(content)

    return timestamp

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

        if discrep_type == 'l2':
            loss_fn = policy_eval.mse_loss
        elif discrep_type == 'max':
            loss_fn = policy_eval.max_loss

        discrepancies = []
        x = y = np.linspace(0, 1, num_ticks)
        for i in range(num_ticks):
            p = x[i]
            for j in range(num_ticks):
                q = y[j]
                pi = np.array([[p, 1 - p], [q, 1 - q], [0, 0]])
                discrepancies.append(loss_fn(pi, value_type))

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
    parser.add_argument('--run_generated', type=str,
        help='name of directory with generated pomdp files located in environment/pomdp_files/generated')
    parser.add_argument('--pomdp_id', default=None, type=int)
    parser.add_argument('--mem_fn_id', default=None, type=int)
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
    parser.add_argument('--generate_pomdps', default=None, nargs=8, type=int,
        help='args: n_pomdps, n_policies, min_n_s, max_n_s, min_n_a, max_n_a, min_n_o, max_n_o; generate pomdp specs and save to environment/pomdp_files/generated/')
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

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        mem_part = 'no_memory'
        if args.use_memory is not None and args.use_memory > 0:
            mem_part = f'memory_{args.use_memory}'
        if args.run_generated:
            name = f'logs/{args.run_generated}.log'
        else:
            name = f'logs/{args.spec}-{mem_part}-{time.time()}.log'
        rootLogger.addHandler(logging.FileHandler(name))

    if args.seed:
        np.random.seed(args.seed)
        jax.random.PRNGKey(args.seed)

    # Run
    if args.generate_pomdps:
        a = args.generate_pomdps
        params = {
            'n_pomdps': a[0],
            'n_policies': a[1],
            'min_n_s': a[2],
            'max_n_s': a[3],
            'min_n_a': a[4],
            'max_n_a': a[5],
            'min_n_o': a[6],
            'max_n_o': a[7]
        }
        timestamp = generate_pomdps(params)

        print(f'Saved generated pomdp files with timestamp: {timestamp}')
    elif args.run_generated:
        run_generated(f'grl/environment/pomdp_files/generated/{args.run_generated}',
                      pomdp_id=args.pomdp_id,
                      mem_fn_id=args.mem_fn_id)
    else:
        # Get POMDP definition
        spec = load_spec(args.spec, args.use_memory)
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

        if args.heatmap:
            heatmap(spec)
        else:
            run_algos(spec, args.method, args.n_random_policies, args.use_grad, args.n_episodes)
