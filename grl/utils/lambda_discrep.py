import jax.numpy as jnp
from typing import Callable
from functools import partial

from grl import AbstractMDP
from grl.utils.loss import discrep_loss
from grl.utils.policy_eval import analytical_pe

def calc_discrep_from_values(td_vals: dict, mc_vals: dict, error_type: str = 'l2'):
    v_diff = td_vals['v'] - mc_vals['v']
    q_diff = td_vals['q'] - mc_vals['q']
    if error_type == 'l2':
        v_discrep = v_diff**2
        q_discrep = q_diff**2
    elif error_type == 'abs':
        v_discrep = jnp.abs(v_diff)
        q_discrep = jnp.abs(q_diff)
    else:
        raise NotImplementedError

    # if weight_discrep:
    #     raise NotImplementedError

    return {'v': v_discrep, 'q': q_discrep}

def lambda_discrep_measures(amdp: AbstractMDP, pi: jnp.ndarray, discrep_loss_fn: Callable = None):
    if discrep_loss_fn is None:
        discrep_loss_fn = partial(discrep_loss, value_type='q', error_type='l2', weight_discrep_by_count=False)

    state_vals, mc_vals, td_vals, _ = analytical_pe(pi, amdp)

    measures = {
        'discrep': discrep_loss_fn(pi, amdp),
        'mc_vals_q': mc_vals['q'],
        'td_vals_q': td_vals['q'],
        'mc_vals_v': mc_vals['v'],
        'td_vals_v': td_vals['v'],
        'state_vals_v': state_vals['v'],
        'state_vals_q': state_vals['q'],
        'p0': amdp.p0.copy()
    }
    return measures
