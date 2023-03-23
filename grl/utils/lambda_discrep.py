import jax.numpy as jnp
from typing import Callable
from functools import partial

from grl import AbstractMDP
from grl.utils.loss import discrep_loss
from grl.utils.policy_eval import analytical_pe

def lambda_discrep_measures(amdp: AbstractMDP, pi: jnp.ndarray, discrep_loss_fn: Callable = None):
    if discrep_loss_fn is None:
        discrep_loss_fn = partial(discrep_loss, value_type='q', error_type='l2', alpha=1.)

    state_vals, _, _, _ = analytical_pe(pi, amdp)
    discrep, mc_vals, td_vals = discrep_loss_fn(pi, amdp)

    measures = {
        'discrep': discrep,
        'mc_vals_q': mc_vals['q'],
        'td_vals_q': td_vals['q'],
        'mc_vals_v': mc_vals['v'],
        'td_vals_v': td_vals['v'],
        'state_vals_v': state_vals['v'],
        'state_vals_q': state_vals['q'],
        'p0': amdp.p0.copy()
    }
    return measures
