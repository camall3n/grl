import jax.numpy as jnp
from jax import nn

from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.utils.pe import analytical_pe, functional_solve_mdp
from grl.memory import functional_memory_cross_product
"""
The following few functions are loss function w.r.t. memory parameters, mem_params.
"""

def mem_diff(value_type: str, mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray,
             T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    _, mc_vals, td_vals = analytical_pe(pi, phi_x, T_x, R_x, p0_x, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return diff, mc_vals, td_vals

def mem_v_l2_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                  R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('v', mem_params, gamma, pi, T, R, phi, p0)
    return (diff**2).mean()

def mem_q_l2_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                  R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('q', mem_params, gamma, pi, T, R, phi, p0)
    diff = diff * pi.T
    return (diff**2).mean()

def mem_v_abs_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                   R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('v', mem_params, gamma, pi, T, R, phi, p0)
    return jnp.abs(diff).mean()

def mem_q_abs_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                   R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('q', mem_params, gamma, pi, T, R, phi, p0)
    diff = diff * pi.T
    return jnp.abs(diff).mean()

"""
The following few functions are loss function w.r.t. policy parameters, pi_params.
"""

def policy_calc_discrep(value_type: str, pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray,
                        R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    pi = nn.softmax(pi_params, axis=-1)
    _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return diff, mc_vals, td_vals, pi

def policy_discrep_v_l2_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                             phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, _ = policy_calc_discrep('v', pi_params, gamma, T, R, phi, p0)
    return (diff**2).mean(), (mc_vals, td_vals)

def policy_discrep_q_l2_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                             phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, pi = policy_calc_discrep('q', pi_params, gamma, T, R, phi, p0)
    diff = diff * pi.T
    return (diff**2).mean(), (mc_vals, td_vals)

def policy_discrep_v_abs_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                              phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, _ = policy_calc_discrep('v', pi_params, gamma, T, R, phi, p0)
    return jnp.abs(diff).mean(), (mc_vals, td_vals)

def policy_discrep_q_abs_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                              phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, pi = policy_calc_discrep('q', pi_params, gamma, T, R, phi, p0)
    diff = diff * pi.T
    return jnp.abs(diff).mean(), (mc_vals, td_vals)

def pg_objective_func(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, phi: jnp.ndarray,
                      p0: jnp.ndarray, R: jnp.ndarray):
    """
    Policy gradient objective function:
    sum_{s_0} p(s_0) v_pi(s_0)
    """
    pi_abs = nn.softmax(pi_params, axis=-1)
    pi_ground = phi @ pi_abs
    occupancy = functional_get_occupancy(pi_ground, T, p0, gamma)

    p_pi_of_s_given_o = get_p_s_given_o(phi, occupancy)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, phi, T, R)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_abs, T_obs_obs, R_obs_obs, gamma)
    p_init_obs = p0 @ phi
    return jnp.dot(p_init_obs, td_v_vals), (td_v_vals, td_q_vals)
