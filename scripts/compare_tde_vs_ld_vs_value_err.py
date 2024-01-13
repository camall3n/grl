import argparse
import logging
import pathlib
from time import time

import numpy as np
import jax
from jax.config import config

from grl.environment import load_pomdp
from grl.environment.policy_lib import get_start_pi
from grl.utils.loss import mstd_err, discrep_loss
from grl.utils.file_system import results_path, numpyify_and_save
from grl.memory import get_memory
from grl.memory_iteration import run_memory_iteration
from grl.utils.mdp import functional_get_occupancy
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

mstd_err(pi_phi, pomdp, value_type='q')
discrep_loss(pi_phi, pomdp, value_type='q')
c_total = float(sum(c_s[:3]))
0.5*0.061765919396285174**2 + 0.5*( (c_s[0]+c_s[1])*(-0.24508320730324495)**2 + c_s[2]*(1.754916792696755**2) ) / c_total
