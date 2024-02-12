import argparse
import logging
import pathlib
from time import time

import numpy as np
import jax
from jax.config import config
from jax.nn import softmax

from grl.environment import load_pomdp
from grl.environment.policy_lib import get_start_pi
from grl.utils.loss import mstd_err, discrep_loss, value_error
from grl.utils.file_system import results_path, numpyify_and_save
from grl.memory import get_memory, memory_cross_product
from grl.memory_iteration import run_memory_iteration
from grl.utils.mdp import functional_get_occupancy
from grl.utils.augment_policy import construct_aug_policy
from grl.vi import td_pe


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

mstd_err(pi_phi, pomdp, value_type='q')[0]
discrep_loss(pi_phi, pomdp, value_type='q')[0]
value_error(pi_phi, pomdp, value_type='q')[0]

mem_hold = get_memory('101')
mem_hold.shape
pomdp_hold = memory_cross_product(mem_hold, pomdp)

mem_flip = get_memory('102')
mem_flip.shape
pomdp_flip = memory_cross_product(mem_flip, pomdp)

mem_set = get_memory('103')
pomdp_set = memory_cross_product(mem_set, pomdp)

n_mem_states = 2
pi_augmented = (
    np.expand_dims(pi_phi, 1)
    .repeat(n_mem_states, 1)
    .reshape(pomdp.observation_space.n * n_mem_states, pomdp.action_space.n)
)

mstd_err(pi_augmented, pomdp_flip, value_type='q')[0]
discrep_loss(pi_augmented, pomdp_flip, value_type='q')[0]
value_error(pi_augmented, pomdp_flip, value_type='q')[0]


mstd_err(pi_augmented, pomdp_set, value_type='q')[0]
discrep_loss(pi_augmented, pomdp_set, value_type='q')[0]
value_error(pi_augmented, pomdp_set, value_type='q')[0]
