import jax.numpy as jnp
from grl import AbstractMDP, PolicyEval
from grl.utils.mdp import functional_get_occupancy

def calc_discrep_from_values(td_vals: dict, mc_vals: dict, error_type: str = 'l2',
                             weight_discrep: bool = False):
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

def lambda_discrep_measures(amdp: AbstractMDP, pi: jnp.ndarray):
    amdp_pe = PolicyEval(amdp)
    pi_ground = amdp.phi @ pi

    state_vals, mc_vals, td_vals, _ = amdp_pe.run(pi)
    pi_occupancy = functional_get_occupancy(pi_ground, amdp)
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
