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
from grl.utils.policy_eval import analytical_pe, lstdq_lambda

#%%

np.set_printoptions(precision=8)

spec = 'example_26a'
seed = 42

np.set_printoptions(precision=8, suppress=True)
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

rand_key = None
np.random.seed(seed)
rand_key = jax.random.PRNGKey(seed)

pomdp, pi_dict = load_pomdp(spec, rand_key)
pomdp.gamma = 0.99999

if 'Pi_phi' in pi_dict and pi_dict['Pi_phi'] is not None:
    pi_phi = pi_dict['Pi_phi'][0]
    print(f'Pi_phi:\n {pi_phi}')

p = 0.1
lds = []
ps = np.linspace(0,1,500)
for p in ps:

    pi_phi = np.array(
        [[p, 1-p],
         [p, 1-p],
         [p, 1-p],
         [p, 1-p]]
    )

    state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)
# state_vals['v']
# state_vals['q']
# mc_vals['v']
# mc_vals['q']
# td_vals['v']
# td_vals['q']

    lds.append(discrep_loss(pi_phi, pomdp, alpha=0)[0])

# value_error(pi_phi, pomdp)

plt.semilogy(ps, lds)
plt.xlabel("Pr(stay)")
plt.ylabel("log LD")
