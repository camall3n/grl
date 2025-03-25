from typing import Union
import jax
from jax import random

# from jax.config import config
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

mpl.rcParams.update(
    {
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
    }
)
np.set_printoptions(precision=8)

seed = 2020
n_samples = 11

rng = random.PRNGKey(seed=seed)
np.random.seed(seed)


def get_policy(spec, p=0.5, q=0.5, r=0.5, a=0.5):
    if spec in ["ld_zero_by_t_projection", "ld_zero_by_mdp"]:
        pi = np.array(
            [
                [p, 1 - p],
                [q, 1 - q],
                [r, 1 - r],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
    elif spec in ["ld_zero_by_r_projection", "ld_zero_by_wr_projection"]:
        pi = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [p, 1 - p],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
    elif spec == "ld_zero_by_k_equality":
        pi = np.array(
            [
                [p, 1 - p],
                [q, 1 - q],
                [q, 1 - q],
                [r, 1 - r],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
    else:
        raise NotImplementedError()
    return pi


def get_random_policy(shape):
    pi = np.random.rand(*shape)
    return pi / pi.sum(axis=1)[:, None]


def get_max_diffs(pi, pomdp):
    s_occupancy = pomdp_get_occupancy(pi, pomdp)
    mc_sa_occupancy = s_occupancy.T[:, None] * pomdp.get_ground_policy(pi)
    W = get_p_s_given_o(pomdp.phi, s_occupancy).T
    td_policy = np.einsum("sw,wa,wt->sta", pomdp.phi, pi, W)
    pi_s = pomdp.get_ground_policy(pi)
    Pi_s = np.eye(len(pi_s))[..., None] * pi_s[None, ...]
    Pi = np.eye(len(pi))[..., None] * pi[None, ...]
    Phi = pomdp.phi
    T = np.moveaxis(pomdp.T, 0, 1)

    _, a = pi_s.shape
    s, o = Phi.shape
    sa = s * a
    oa = o * a
    # plt.imshow(Phi @ W)

    I_a = np.eye(a)
    W_A = np.kron(W, I_a).reshape(o, a, s, a)
    Phi_A = np.kron(Phi, I_a).reshape(s, a, o, a)
    I_sa = np.eye(sa).reshape(s, a, s, a)
    I_oa = np.eye(oa).reshape(o, a, o, a)

    def dot(x, *args):
        while args:
            y, args = args[0], args[1:]
            x = np.tensordot(x, y, axes=1)
        return x

    def ddot(x, *args):
        while args:
            y, args = args[0], args[1:]
            # x = np.einsum('ijkl,klmn->ijmn', x, y)
            x = np.tensordot(x, y, axes=2)
        return x

    def dpow(a, exp):
        x, *rest = [a] * exp
        return ddot(x, *rest)

    # Ψ^{SA} using MC policy spreading tensor
    MC_SR_sasa = np.linalg.tensorinv(I_sa - pomdp.gamma * dot(T, Pi_s))

    # Ψ^{SA} using TD policy spreading tensor
    TD_SR_sasa = np.linalg.tensorinv(I_sa - pomdp.gamma * dot(T, td_policy))

    # Ψ^{ΩA} using effective MDP transition fn
    TD_SF_yellow_oaoa = np.linalg.tensorinv(I_oa - pomdp.gamma * dot(W, T, Phi, Pi))
    TD_SF_blue_oaoa = dot(W, ddot(TD_SR_sasa, Phi_A))

    # Ψ^{ΩA} using (W Ψ^{SA} Φ), for the Ψ^{SA} with the MC policy spreading tensor
    MC_SF_oaoa = dot(W, ddot(MC_SR_sasa, Phi_A))

    # pomdp_mdp_predictions = [
    #     (dot(W, I_sa), ddot(I_oa, W_A)),
    #     (dot(W, T_sasa), ddot(T_oaoa, W_A)),
    # ]
    # for i in range(2, 30):
    #     pomdp_mdp_predictions.append(
    #         (dot(W, dpow(T_sasa, i)), ddot(dpow(T_oaoa, i), W_A))
    #     )
    #
    # for pred_pomdp, pred_mdp in pomdp_mdp_predictions:
    #     assert np.allclose(pred_pomdp, pred_mdp)
    #     print('.', end='')
    # print()

    lambda_0 = 0.0
    lambda_1 = 1.0

    def get_K(Pi_s, td_policy, lambda_):
        return lambda_ * Pi_s + (1 - lambda_) * td_policy

    K0 = get_K(Pi_s, td_policy, lambda_0)
    K1 = get_K(Pi_s, td_policy, lambda_1)
    T_sasa_0 = dot(T, K0)
    T_sasa_1 = dot(T, K1)
    R_sa = np.einsum("ast,ast->sa", pomdp.T, pomdp.R)

    # TD(λ) versions (identical to the above for λ in {0, 1})
    SR_0 = np.linalg.tensorinv(I_sa - pomdp.gamma * T_sasa_0)
    SR_1 = np.linalg.tensorinv(I_sa - pomdp.gamma * T_sasa_1)

    Q0_sa = ddot(SR_0, R_sa)
    Q1_sa = ddot(SR_1, R_sa)

    # ΩA x SA
    SF_0 = W @ SR_0
    SF_1 = W @ SR_1

    Q0_wa = W @ Q0_sa
    Q1_wa = W @ Q1_sa

    diffs = {
        "∆ K": np.max(np.abs(K0 - K1)),
        "∆ SR_sasa": np.max(np.abs(SR_0 - SR_1)),
        "∆ Q_sa": np.max(np.abs(Q0_sa - Q1_sa)),
        "∆ Q_wa": np.max(np.abs(Q0_wa - Q1_wa)),
        # "Δ SR_sasa": np.max(np.abs(MC_SR_sasa - TD_SR_sasa)),
        # "Δ SF_oasa": np.max(np.abs(MC_SF_oasa - TD_SF_oasa)),
        "Δ SF_oasa": np.max(np.abs(SF_0 - SF_1)),
        "Δ SF_oaoa_yellow_red": np.max(np.abs(MC_SF_oaoa - TD_SF_yellow_oaoa)),
        "Δ SF_oaoa_blue_red": np.max(np.abs(MC_SF_oaoa - TD_SF_blue_oaoa)),
        "Δ SF_TD_yellow_blue": np.max(np.abs(TD_SF_blue_oaoa - TD_SF_yellow_oaoa)),
    }
    return diffs


specs_and_n_params = {
    "ld_zero_by_mdp": 3,
    "ld_zero_by_k_equality": 3,
    "ld_zero_by_t_projection": 3,
    "ld_zero_by_r_projection": 1,
    "ld_zero_by_wr_projection": 1,
    "tiger-alt-start": 0,
    "network": 0,
    "tmaze_5_two_thirds_up": 0,
    "example_7": 0,
    "4x3.95": 0,
    "cheese.95": 0,
    "network": 0,
    "shuttle.95": 0,
    "paint.95": 0,
    "hallway": 0,
    "bridge-repair": 0,
}
# %%
data = []
ps = np.linspace(0, 1, n_samples)
for spec, n_params in specs_and_n_params.items():
    pomdp, info = load_pomdp(spec)
    if n_params > 0:
        all_ps = np.reshape(np.meshgrid(*[ps] * n_params), (n_params, -1)).T
    else:  # n_params == 0
        all_ps = range(10)
    for probs in tqdm(all_ps):
        if n_params > 0:
            pi = get_policy(spec, *probs)
        else:
            n_o = pomdp.phi.shape[1]
            n_a = pomdp.T.shape[0]
            pi = get_random_policy((n_o, n_a))
        diffs = get_max_diffs(pi, pomdp)
        diffs["spec"] = spec
        data.append(diffs)

spec_order = lambda specs: [
    list(specs_and_n_params.keys()).index(spec) for spec in specs
]

result = (
    pd.DataFrame(data)
    .groupby(["spec"])
    .max()
    .round(4)[
        [
            "∆ K",
            "∆ SR_sasa",
            "∆ Q_sa",
            "∆ Q_wa",
            "Δ SF_oasa",
            "Δ SF_oaoa_yellow_red",
            "Δ SF_oaoa_blue_red",
            "Δ SF_TD_yellow_blue",
        ]
    ]
    .sort_index(key=spec_order)
)

print(result)
with open("output.txt", "w") as f:
    f.write(result.to_string())
