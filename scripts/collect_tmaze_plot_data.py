import argparse
from jax import jit
import numpy as np
import pandas as pd

from jax.nn import softmax
from jax.config import config
from tqdm import tqdm
from pathlib import Path

from grl.environment import load_spec
from grl.memory import get_memory
from grl.mdp import MDP, POMDP
from grl.utils.math import reverse_softmax
from grl.utils.loss import mem_discrep_loss

from definitions import ROOT_DIR

def calc_init_obs_dist(mem_fns: np.ndarray, action_idx: int = 2):
    ab = np.stack((mem_fns[:, action_idx, 0, 0, 0], mem_fns[:, action_idx, 1, 0, 0]), axis=-1)
    first_min = np.linalg.norm(ab - np.array([1, 0]), axis=-1, ord=2)
    second_min = np.linalg.norm(ab - np.array([0, 1]), axis=-1, ord=2)
    return np.minimum(first_min, second_min)

if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n',
        default=100,
        type=int,
        help='How many points for corridor do we linspace over? (total points is squared of this)')
    parser.add_argument(
        '--m',
        default=10,
        type=int,
        help='How many points for init_obs do we linspace over? (total points is squared of this)')
    args = parser.parse_args()

    df_path = Path(ROOT_DIR, 'results', 'analytical_tmaze_plot_data.pkl')

    spec = load_spec('tmaze_5_two_thirds_up')

    mem_params = get_memory('0', n_mem_states=2)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])

    # corridor values
    ps = np.linspace(0., 1., num=args.n)

    # init obs values
    qs = np.linspace(0., 1., num=args.m)

    all_ps_qs = np.array(np.meshgrid(qs, qs, ps, ps)).T.reshape(-1, 4)

    n_total_mem_funcs = all_ps_qs.shape[0]
    all_mem_funcs = np.expand_dims(softmax(mem_params, axis=-1), 0).repeat(n_total_mem_funcs,
                                                                           axis=0)

    # init obs
    all_mem_funcs[:, 2, 0, 0, 0] = all_ps_qs[:, 0]
    all_mem_funcs[:, 2, 0, 0, 1] = 1 - all_ps_qs[:, 0]

    all_mem_funcs[:, 2, 1, 0, 0] = all_ps_qs[:, 1]
    all_mem_funcs[:, 2, 1, 0, 1] = 1 - all_ps_qs[:, 1]

    # corridor
    all_mem_funcs[:, 2, 2, 0, 0] = all_ps_qs[:, 2]
    all_mem_funcs[:, 2, 2, 0, 1] = 1 - all_ps_qs[:, 2]
    all_mem_funcs[:, 2, 2, 1, 0] = all_ps_qs[:, 3]
    all_mem_funcs[:, 2, 2, 1, 1] = 1 - all_ps_qs[:, 3]

    all_ys = calc_init_obs_dist(all_mem_funcs)

    pi = spec['Pi_phi'][0]
    pi_x = pi.repeat(all_mem_funcs.shape[-1], axis=0)

    mem_funcs = all_mem_funcs
    ys = all_ys
    p1s = all_ps_qs[:, 2]
    p2s = all_ps_qs[:, 3]

    discreps = np.zeros(mem_funcs.shape[0])

    # optimal_mem1 = get_memory("16")
    # optimal_mem2 = get_memory("18")
    #
    # discrep1 = calc_discrep(optimal_mem1, spec['gamma'], pi_x, spec['T'], spec['R'], spec['phi'], spec['p0'])
    # print()
    # discrep2 = calc_discrep(optimal_mem2, spec['gamma'], pi_x, spec['T'], spec['R'], spec['phi'], spec['p0'])

    calc_discrep = jit(mem_discrep_loss)
    for i, mem_func in enumerate(tqdm(mem_funcs)):
        discreps[i] = calc_discrep(reverse_softmax(mem_func), spec['gamma'], pi_x, spec['T'],
                                   spec['R'], spec['phi'], spec['p0'])

    df = pd.DataFrame({'y': ys, 'p1': p1s, 'p2': p2s, 'D': discreps})
    df.to_pickle(df_path)

    print(f"Saved {df.shape[0]} rows to {df_path}.")
