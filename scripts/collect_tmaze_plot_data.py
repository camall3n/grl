import argparse
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax.nn import softmax
from jax.config import config
from jax import jit
from tqdm import tqdm
from functools import partial
from pathlib import Path

from grl.memory import functional_memory_cross_product
from grl.environment import load_spec
from grl.environment.memory_lib import get_memory
from grl.utils.lambda_discrep import calc_discrep_from_values
from grl.policy_eval import analytical_pe
from grl.utils.math import reverse_softmax
from grl.utils.loss import mem_q_abs_loss
from definitions import ROOT_DIR

# @partial(jit, static_argnames=['gamma'])
# def calc_discrep(
#         pi_x: jnp.ndarray,
#         T: jnp.ndarray, T_mem: jnp.ndarray, phi: jnp.ndarray,
#         R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
#     # T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
#     # state_vals, mc_vals, td_vals, _ = analytical_pe(pi_x, phi_x, T_x, R_x, p0_x, gamma)
#     # discreps = calc_discrep_from_values(td_vals, mc_vals, error_type='abs')
#     # return discreps['q'].sum()
#     return mem_q_abs_loss()

def calc_init_obs_dist(mem_fns: np.ndarray, action_idx: int = 2):
    ab = np.stack((mem_fns[:, action_idx, 0, 0, 0], mem_fns[:, action_idx, 1, 0, 0]), axis=-1)
    first_min = np.linalg.norm(ab - np.array([1, 0]), axis=-1, ord=2)
    second_min = np.linalg.norm(ab - np.array([0, 1]), axis=-1, ord=2)
    return np.minimum(first_min, second_min)


if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=100, type=int,
                        help='How many points for corridor do we linspace over? (total points is squared of this)')
    parser.add_argument('--m', default=10, type=int,
                        help='How many points for init_obs do we linspace over? (total points is squared of this)')
    args = parser.parse_args()

    df_path = Path(ROOT_DIR, 'results', 'analytical_tmaze_plot_data.pkl')

    spec = load_spec('tmaze_5_two_thirds_up',
                     memory_id=0,
                     n_mem_states=2)

    @partial(jit, static_argnames=['gamma'])
    def calc_discrep(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                     R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
        return mem_q_abs_loss(mem_params, gamma, pi, T, R, phi, p0)

    # corridor values
    ps = np.linspace(0., 1., num=args.n)

    # init obs values
    qs = np.linspace(0., 1., num=args.m)

    all_ps_qs = np.array(np.meshgrid(qs, qs, ps, ps)).T.reshape(-1, 4)

    n_total_mem_funcs = all_ps_qs.shape[0]
    all_mem_funcs = np.expand_dims(softmax(spec['mem_params'], axis=-1), 0).repeat(n_total_mem_funcs, axis=0)

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

    # TODO: change this, this filters only y == 0
    # mask = all_ys == 0
    # mem_funcs = all_mem_funcs[mask]
    # ys = all_ys[mask]
    # p1s = all_corridor_mems[mask][:, 0, 0]
    # p2s = all_corridor_mems[mask][:, 1, 0]

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


    for i, mem_func in enumerate(tqdm(mem_funcs)):
        discreps[i] = calc_discrep(reverse_softmax(mem_func), spec['gamma'], pi_x, spec['T'], spec['R'], spec['phi'], spec['p0'])

    df = pd.DataFrame({'y': ys, 'p1': p1s, 'p2': p2s, 'D': discreps})
    df.to_pickle(df_path)

    print(f"Saved {df.shape[0]} rows to {df_path}.")

