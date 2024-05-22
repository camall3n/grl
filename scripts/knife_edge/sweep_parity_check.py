import argparse
import logging
import pathlib
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import jax
from jax.config import config
from jax.nn import softmax
from jax import random
from jax import tree_map
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.environment.policy_lib import get_start_pi
from grl.utils.loss import mstd_err, discrep_loss, value_error
from grl.utils.file_system import results_path, numpyify_and_save
from grl.memory import get_memory, memory_cross_product
from grl.memory_iteration import run_memory_iteration
from grl.utils.math import reverse_softmax
from grl.utils.mdp import functional_get_occupancy
from grl.utils.policy import construct_aug_policy, get_unif_policies
from grl.vi import td_pe
from grl.utils.policy_eval import analytical_pe, lstdq_lambda

#%%

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

spec = 'parity_check'
seed = 42

np.set_printoptions(precision=8, suppress=True)
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

rand_key = None
np.random.seed(seed)
rand_key = jax.random.PRNGKey(seed)

pomdp, pi_dict = load_pomdp(spec, rand_key)
pomdp.gamma = 0.9
pomdp.phi
if 'Pi_phi' in pi_dict and pi_dict['Pi_phi'] is not None:
    pi_phi = pi_dict['Pi_phi'][0]
    # print(f'Pi_phi:\n {pi_phi}')

#%%
lds = []
ps = np.linspace(0, 1, 500)
for p in tqdm(ps):
    pi_phi = np.array([
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [p, (1-p),     0,     0],
        [1,     0,     0,     0.],
    ])

    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='p', y='ld')
plt.xlabel(r'Junction $\uparrow$ (vs. $\downarrow$) probability')
plt.ylabel("")
plt.title('Mean Squared Lambda Discrepancy')
plt.show()

#%%
lds = []
ps = np.linspace(0, 1, 500)
for p in tqdm(ps):
    pi_phi = np.array([
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     p, (1-p)],
        [1,     0,     0,     0.],
    ])

    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='p', y='ld')
plt.xlabel(r'Junction $\uparrow$ (vs. $\downarrow$) probability')
plt.ylabel("")
plt.title('Mean Squared Lambda Discrepancy')
plt.show()

#%%
T = pomdp.T

fig, axes = plt.subplots(2, 4, figsize=(8, 5))
fig.suptitle(r'Teleport transition dynamics (top: $\epsilon=0$, bottom: $\epsilon=0.5$)')
actions = [r'$\uparrow$', r'$\downarrow$', r'$\rightarrow$', r'$\leftarrow$']
for t, ax, a in zip(T, axes[0], actions):
    ax.imshow(t)
    ax.set_title(a)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$s'$")
    ax.set_xlabel(r"$s$")

def modify_transitions(T, epsilon=0.1):
    Teps = np.ones_like(T)
    Teps[:, -1, :] = 0
    Teps[:, :, -1] = 0
    Teps[:, -1,-1] = 1
    Teps /= Teps.sum(-1, keepdims=True)

    # return Teps

    return (1-epsilon)*T + epsilon*Teps

Tmod = modify_transitions(T, epsilon=0.5)

for t, ax, a in zip(Tmod, axes[1], actions):
    ax.imshow(t, vmin=0, vmax=1)
    ax.set_title(a)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$s'$")
    ax.set_xlabel(r"$s$")

plt.tight_layout()
plt.show()

#%%
pi_phi = pi_dict['Pi_phi'][0]
lds = []
ps = np.linspace(0,1,500)
T = pomdp.T
for p in tqdm(ps):
    pomdp.T = modify_transitions(T, epsilon=p)
    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'eps': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
pomdp.T = T
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='eps', y='ld')
plt.xlabel(r'Random teleport probability ($\epsilon$)')
plt.ylabel("")
plt.title(r'Mean Squared Lambda Discrepancy')
plt.show()

#%%
rand_key = random.PRNGKey(seed)
lds = []
for i in range(500):
    rand_key, pi_key = random.split(rand_key)
    pi = get_unif_policies(pi_key, (pomdp.observation_space.n, pomdp.action_space.n), 1)[0]
    state_vals, mc_vals, td_vals, info = analytical_pe(pi, pomdp)

    lds.append({'i': i, 'ld': discrep_loss(pi, pomdp, alpha=0)[0].item()})
data = pd.DataFrame(lds)
data['log ld'] = np.log10(data['ld'])
sns.histplot(data=data, x='log ld')
plt.xlabel(r'$\log_{10}$(mean squared lambda discrepancy)')
plt.ylabel(r'Number of policies')

#%%
pi_phi = pi_dict['Pi_phi'][0]
lds = []
orig_p0 = pomdp.p0
for i in range(500):
    rand_key, p0_key = random.split(rand_key)

    # start at the beginning of the maze
    p0 = get_unif_policies(p0_key, (4,), 1)[0]
    p0 = np.concatenate((p0, np.zeros(pomdp.state_space.n-4)))

    # # start at any non-terminal state
    # p0 = get_unif_policies(p0_key, (pomdp.state_space.n-1,), 1)[0]
    # p0 = np.concatenate((p0, [0]))

    pomdp.p0 = p0
    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'i': i, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
pomdp.p0 = orig_p0
data = pd.DataFrame(lds)
data['log ld'] = np.log10(data['ld'])
sns.histplot(data=data, x='log ld')
plt.xlabel(r'$\log_{10}$(mean squared lambda discrepancy)')
plt.ylabel(r'Number of starting distributions')

#%%
lds = []
gammas = np.linspace(0, 1, 500)
orig_gamma = pomdp.gamma
for g in tqdm(gammas):
    pomdp.gamma = g
    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'gamma': g, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
pomdp.gamma = orig_gamma
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='gamma', y='ld')
plt.xlabel(r'Discount factor ($\gamma$)')
plt.ylabel("")
plt.title('Mean Squared Lambda Discrepancy')
plt.show()

#%%

lds = []
lambdas = np.linspace(0, 1, 10)
state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)
for l0 in tqdm(lambdas):
    for l1 in lambdas:
        lds.append({'l0': l0, 'l1': l1, 'ld': discrep_loss(pi_phi, pomdp, lambda_0=l0, lambda_1=l1, alpha=0)[0].item()})
data = pd.DataFrame(lds)
data['log ld'] = np.log10(data['ld'])
#%%
sns.heatmap(data.pivot(index="l1", columns="l0", values="ld"), square=True)
plt.xticks([0, 10], [0, 1])
plt.yticks([0, 10], [0, 1])
plt.xlabel(r'$\lambda_0$')
plt.ylabel(r'$\lambda_1$')
plt.gca().invert_yaxis()
plt.title('Mean Squared Lambda Discrepancy')
plt.show()

#%%
lds = []
ps = np.linspace(0, 1, 500)
for p in tqdm(ps):
    pi_phi = np.array([
        [p,     0, (1-p),     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [2/3, 1/3,     0,     0],
        [1,     0,     0,     0.],
    ])

    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='p', y='ld')
plt.semilogy()
plt.xlabel(r'$\Pr(\textsc{stay}|\textsc{blue})$')
plt.ylabel("")
plt.title('Lambda Discrepancy')
plt.show()

#%%
pi_phi = pi_dict['Pi_phi'][0]
lds = []
orig_p0 = pomdp.p0
ps = np.linspace(0, 1, 20000)
for p in tqdm(ps):
    # start at the beginning of the maze
    unif_others = (1-p)/3 * np.ones(3)
    p0 = np.concatenate(([p], unif_others, np.zeros(pomdp.state_space.n-4)))

    # # start at any non-terminal state
    # p0 = get_unif_policies(p0_key, (pomdp.state_space.n-1,), 1)[0]
    # p0 = np.concatenate((p0, [0]))

    pomdp.p0 = p0
    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
pomdp.p0 = orig_p0
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='p', y='ld')
plt.xlabel(r'Probability of initializing to \textsc{red} $\rightarrow$ \textsc{pink} path')
plt.semilogy()
plt.ylabel(r'')
plt.title('Lambda Discrepancy')
plt.show()


#%%
fig, ax = plt.subplots(1, 2, figsize=(6,2))

lds = []
ps = np.linspace(0, 1, 500)
for p in tqdm(ps):
    pi_phi = np.array([
        [p,     0, (1-p),     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [0,     0,     1,     0],
        [2/3, 1/3,     0,     0],
        [1,     0,     0,     0.],
    ])

    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='p', y='ld', ax=ax[0], label=r'$\Pr(\textsc{stay}|\textsc{blue})$')
ax[0].semilogy()
# ax[0].set_xlabel()
ax[0].set_ylabel(r'$\lambda$-discrepancy')
ax[0].legend(loc='lower right')


pi_phi = pi_dict['Pi_phi'][0]
lds = []
orig_p0 = pomdp.p0
ps = np.linspace(0, 1, 20000)
for p in tqdm(ps):
    # start at the beginning of the maze
    unif_others = (1-p)/3 * np.ones(3)
    p0 = np.concatenate(([p], unif_others, np.zeros(pomdp.state_space.n-4)))

    # # start at any non-terminal state
    # p0 = get_unif_policies(p0_key, (pomdp.state_space.n-1,), 1)[0]
    # p0 = np.concatenate((p0, [0]))

    pomdp.p0 = p0
    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)

    lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item()})
pomdp.p0 = orig_p0
data = pd.DataFrame(lds)
sns.lineplot(data=data, x='p', y='ld', ax=ax[1], label=r'$p_0(\textsc{red} \rightarrow \textsc{pink})$')
ax[1].semilogy()
ax[0].set_xlabel(r'Probability')
ax[1].set_xlabel(r'Probability')
ax[1].set_ylabel(r'$\lambda$-discrepancy')
ax[1].legend(loc='lower left')
plt.tight_layout()
plt.savefig('nice-sweeps.png')
plt.show()
