import jax.numpy as jnp
from grl import AbstractMDP, PolicyEval
from grl.utils.mdp import functional_get_occupancy

def lambda_discrep_measures(amdp: AbstractMDP, pi: jnp.ndarray):
    amdp_pe = PolicyEval(amdp)
    pi_ground = amdp.phi @ pi

    state_vals, mc_vals, td_vals = amdp_pe.run(pi)
    pi_occupancy = functional_get_occupancy(pi_ground, amdp.T, amdp.p0, amdp.gamma)
    pr_oa = (pi_occupancy @ amdp.phi * pi.T)
    discrep = {
        'v': (mc_vals['v'] - td_vals['v'])**2,
        'q': (mc_vals['q'] - td_vals['q'])**2,
        'mc_vals_q': mc_vals['q'],
        'td_vals_q': td_vals['q'],
        'mc_vals_v': mc_vals['v'],
        'td_vals_v': td_vals['v'],
        'state_vals_v': state_vals['v'],
        'state_vals_q': state_vals['q'],
        'p0': amdp.p0.copy()
    }
    discrep['q_sum'] = (discrep['q'] * pr_oa).sum()
    return discrep
