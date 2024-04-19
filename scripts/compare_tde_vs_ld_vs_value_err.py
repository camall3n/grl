import argparse
import logging
import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import jax
from jax.config import config
from jax.nn import softmax

from grl.environment import load_pomdp
from grl.environment.policy_lib import get_start_pi
from grl.utils.loss import mstd_err, discrep_loss, value_error
from grl.utils.file_system import results_path, numpyify_and_save
from grl.memory import get_memory, memory_cross_product
from grl.memory_iteration import run_memory_iteration
from grl.utils.math import reverse_softmax
from grl.utils.mdp import functional_get_occupancy
from grl.utils.policy import construct_aug_policy
from grl.vi import td_pe

#%%

np.set_printoptions(precision=8)

spec = 'example_20'
seed = 42

np.set_printoptions(precision=8, suppress=True)
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

rand_key = None
np.random.seed(seed)
rand_key = jax.random.PRNGKey(seed)

pomdp, pi_dict = load_pomdp(spec, rand_key)

print(f'spec:\n {spec}\n')
print(f'T:\n {pomdp.T}')
print(f'R:\n {pomdp.R}')
print(f'gamma: {pomdp.gamma}')
print(f'p0:\n {pomdp.p0}')
print(f'phi:\n {pomdp.phi}')

if 'Pi_phi' in pi_dict and pi_dict['Pi_phi'] is not None:
    pi_phi = pi_dict['Pi_phi'][0]
    print(f'Pi_phi:\n {pi_phi}')

p = 0.5
pi_phi = np.array(
    [[p, 1-p],
     [p, 1-p]]
)

td_v, td_q = td_pe(pi_phi, pomdp)
c_s = functional_get_occupancy(pomdp.get_ground_policy(pi_phi), pomdp.base_mdp)
print('td_q:', td_q)
print('td_v:', td_v)

y = pomdp.gamma
q_up = td_q[0,0]
q_dn = td_q[1,0]
v = td_v[0]

print('(r+\gamma q_up) - q_up:', 0+y*q_up - q_up)
print('(r+\gamma q_up) - q_up:', 0+y*q_dn - q_up)
print('r_bad - q_dn:', (-1) - q_dn)
print('r_good - q_dn:', 1 - q_dn)
print('c_s:', c_s)

mem_hold = get_memory('101')
mem_hold.shape
pomdp_hold = memory_cross_product(mem_hold, pomdp)

mem_flip = get_memory('102')
mem_flip.shape
pomdp_toggle = memory_cross_product(mem_flip, pomdp)

mem_set = get_memory('103')
pomdp_set = memory_cross_product(mem_set, pomdp)

n_mem_states = 2
pi_augmented = (
    np.expand_dims(pi_phi, 1)
    .repeat(n_mem_states, 1)
    .reshape(pomdp.observation_space.n * n_mem_states, pomdp.action_space.n)
)

mem_perfect = np.array([
    [0, 1, 0.],
    [0, 0, 1.],
    [1, 0, 0.],
])
mem_perfect = np.stack([mem_perfect, mem_perfect])
mem_perfect = np.stack([mem_perfect, mem_perfect])
mem_perfect = reverse_softmax(mem_perfect)

resolved_mdp = memory_cross_product(mem_perfect, pomdp)
n_mem_states = 3
pi_perfect = (
    np.expand_dims(pi_phi, 1)
    .repeat(n_mem_states, 1)
    .reshape(pomdp.observation_space.n * n_mem_states, pomdp.action_space.n)
)

mstde = np.array([
    mstd_err(pi_phi, pomdp)[0],
    mstd_err(pi_augmented, pomdp_hold)[0],
    mstd_err(pi_augmented, pomdp_toggle)[0],
    mstd_err(pi_augmented, pomdp_set)[0],
    mstd_err(pi_perfect, resolved_mdp)[0],
])

alpha=0
ld = np.array([
    discrep_loss(pi_phi, pomdp, alpha=alpha)[0],
    discrep_loss(pi_augmented, pomdp_hold, alpha=alpha)[0],
    discrep_loss(pi_augmented, pomdp_toggle, alpha=alpha)[0],
    discrep_loss(pi_augmented, pomdp_set, alpha=alpha)[0],
    discrep_loss(pi_perfect, resolved_mdp, alpha=alpha)[0],
])

val_err = np.array([
    value_error(pi_phi, pomdp)[0],
    value_error(pi_augmented, pomdp_hold)[0],
    value_error(pi_augmented, pomdp_toggle)[0],
    value_error(pi_augmented, pomdp_set)[0],
    value_error(pi_perfect, resolved_mdp)[0],
])

labels = [
    'NONE',
    'HOLD',
    'TOGGLE',
    'SET',
    'PERFECT',
]

#%%
x = np.arange(5)
dx = 0.2
plt.bar(x-dx, mstde, color='C2', width=dx, label='MSTDE')
plt.bar(x, val_err, color='C0', width=dx, label='Value Error')
plt.bar(x+dx, ld, color='C1', width=dx, label='Lambda Discrep')
plt.xticks(x, labels)
plt.legend()
plt.ylim([0, 0.35])
plt.xlabel('Memory Function')
plt.ylabel('Error')
plt.show()
