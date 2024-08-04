from typing import Union
import jax
from jax import random
from jax.config import config
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from grl.environment import load_pomdp
from grl.memory.lib import get_memory
from grl.utils.loss import mem_discrep_loss, discrep_loss
from grl.utils.mdp import pomdp_get_occupancy, get_p_s_given_o, MDP, POMDP
from grl.utils.policy_eval import analytical_pe

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans serif"],
    "font.monospace": ["Computer Modern Typewriter"],
    "axes.labelsize": 12,  # LaTeX default is 10pt
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
np.set_printoptions(precision=8)

seed = 2020
n_samples = 21

rng = random.PRNGKey(seed=seed)
np.random.seed(seed)

def get_policy(spec, p, q=0.5, r=0.5, a=0.5):
    if spec == 'ld_zero_by_t_projection':
        pi = np.array([
            [p, 1-p],
            [q, 1-q],
            [r, 1-r],
            [.5, .5],
            [.5, .5],
        ])
    elif spec in ['ld_zero_by_r_projection', 'ld_zero_by_w_projection']:
        pi = np.array([
            [.5, .5],
            [.5, .5],
            [.5, .5],
            [.5, .5],
            [p, 1-p],
            [.5, .5],
            [.5, .5],
        ])
    elif spec == 'ld_zero_by_k_equality':
        pi = np.array([
            [p, 1-p],
            [p, 1-p],
            [q, 1-q],
            [r, 1-r],
            [.5, .5],
            [.5, .5],
        ])
    else:
        raise NotImplementedError()
    return pi


def get_max_diffs(pi, pomdp):
    s_occupancy = pomdp_get_occupancy(pi, pomdp)
    mc_sa_occupancy = s_occupancy.T[:,None] * pomdp.get_ground_policy(pi)
    W = get_p_s_given_o(pomdp.phi, s_occupancy).T
    td_policy = np.einsum("sw,wa,wt->sta", pomdp.phi, pi, W)
    pi_s = pomdp.get_ground_policy(pi)
    Pi_s = np.eye(len(pi_s))[..., None] * pi_s[None,...]
    td_policy.shape
    Pi_s.shape

    lambda_0 = 0.0
    lambda_1 = 1.0
    def get_K(Pi_s, td_policy, lambda_):
        return lambda_ * Pi_s + (1-lambda_) * td_policy
    K0 = get_K(Pi_s, td_policy, lambda_0)
    K1 = get_K(Pi_s, td_policy, lambda_1)
    n_a, n_s, _ = pomdp.T.shape
    n_sa = n_a * n_s
    T_sasa_0 = np.reshape(np.einsum('aij,jkb->iakb', pomdp.T, K0), (n_sa, n_sa))
    T_sasa_1 = np.reshape(np.einsum('aij,jkb->iakb', pomdp.T, K1), (n_sa, n_sa))
    R_sa = np.reshape(np.einsum("ast,ast->sa", pomdp.T, pomdp.R), n_sa)
    I = np.eye(len(R_sa))

    Q0_sa = np.reshape(np.linalg.solve((I - pomdp.gamma * T_sasa_0), R_sa), (n_s, n_a))
    Q1_sa = np.reshape(np.linalg.solve((I - pomdp.gamma * T_sasa_1), R_sa), (n_s, n_a))

    Q0_wa = W @ Q0_sa
    Q1_wa = W @ Q1_sa

    diffs = {
        '∆ K': np.max(np.abs(K0 - K1)),
        '∆ T_sasa': np.max(np.abs(T_sasa_0 - T_sasa_1)),
        '∆ Q_sa': np.max(np.abs(Q0_sa - Q1_sa)),
        '∆ Q_wa': np.max(np.abs(Q0_wa - Q1_wa)),
    }
    return diffs

spec = 'ld_zero_by_t_projection'

specs_and_n_params = {
    'ld_zero_by_k_equality': 3,
    'ld_zero_by_t_projection': 3,
    'ld_zero_by_r_projection': 1,
    'ld_zero_by_w_projection': 1,
}

data = []
ps = np.linspace(0, 1, n_samples)
for spec, n_params in specs_and_n_params.items():
    pomdp, info = load_pomdp(spec)
    all_ps = np.reshape(np.meshgrid(*[ps]*n_params), (n_params, -1)).T
    for probs in tqdm(all_ps):
        pi = get_policy(spec, *probs)
        pomdp.phi
        diffs = get_max_diffs(pi, pomdp)
        diffs['spec'] = spec
        data.append(diffs)

pd.DataFrame(data).groupby(['spec']).max().round(4)
