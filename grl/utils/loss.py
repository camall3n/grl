import jax.numpy as jnp
from jax import nn, lax, jit
from functools import partial

from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.utils.policy_eval import analytical_pe, functional_solve_mdp
from grl.memory import functional_memory_cross_product
"""
The following few functions are loss function w.r.t. memory parameters, mem_params.
"""

def mem_diff(value_type: str, mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray,
             T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    _, mc_vals, td_vals, _ = analytical_pe(pi, phi_x, T_x, R_x, p0_x, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return diff, mc_vals, td_vals

def weighted_mem_q_abs_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray,
             T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    """
    TODO: Test this?
    """
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    _, mc_vals, td_vals, info = analytical_pe(pi, phi_x, T_x, R_x, p0_x, gamma)
    c_s = info['occupancy']
    c_o = c_s @ phi_x
    p_o = c_o / c_o.sum()
    weight = lax.stop_gradient(pi * p_o[:, None]).T
    diff = jnp.abs(mc_vals['q'] - td_vals['q'])
    loss = diff * weight
    return loss.sum()

@partial(jit, static_argnames=['value_type', 'error_type', 'weight_discrep_by_count', 'gamma'])
def discrep_loss(value_type: str, error_type: str, weight_discrep_by_count: bool, gamma: float, # initialize static args
                 pi: jnp.ndarray, phi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray):  # non-state args
    _, mc_vals, td_vals, info = analytical_pe(pi, phi, T, R, p0, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]

    weight = pi.T if value_type == 'q' else jnp.ones_like(diff)
    if weight_discrep_by_count:
        c_s = info['occupancy']
        c_o = c_s @ phi
        p_o = c_o / c_o.sum()
        weight = lax.stop_gradient(pi * p_o[:, None]).T
        if value_type == 'v':
            weight = weight.sum(axis=0)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    # TODO: CHANGE THIS
    loss = (weight * unweighted_err).mean()
    return loss

def mem_discrep_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray,  # input non-static arrays
                     phi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray,
                     value_type: str, error_type: str, weight_discrep: bool):  # initialize with partial
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    loss = discrep_loss(value_type, error_type, weight_discrep, gamma,
                        pi, phi_x, T_x, R_x, p0_x)
    return loss

"""
The following few functions are loss function w.r.t. policy parameters, pi_params.
"""

def policy_discrep_loss(pi_params: jnp.ndarray, gamma: float,
                        phi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray,
                        value_type: str, error_type: str, weight_discrep: bool):  # args initialize with partial
    pi = nn.softmax(pi_params, axis=-1)
    loss = discrep_loss(value_type, error_type, weight_discrep, gamma,
                        pi, phi, T, R, p0)
    return loss

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

def mem_abs_td_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                    R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    """
    Absolute TD error loss.
    This is an upper bound on absolute lambda discrepancy.
    """
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)

    # observation policy, but expanded over states
    pi_state = phi_x @ pi
    occupancy = functional_get_occupancy(pi_state, T_x, p0_x, gamma)

    p_pi_of_s_given_o = get_p_s_given_o(phi_x, occupancy)

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, phi_x, T_x, R_x)
    td_v_vals, td_q_vals = functional_solve_mdp(pi, T_obs_obs, R_obs_obs, gamma)
    td_vals = {'v': td_v_vals, 'q': td_q_vals}

    # Get starting obs distribution
    obs_p0_x = phi_x * p0_x
    # based on our TD model, get our observation occupancy
    obs_occupancy = functional_get_occupancy(pi, T_obs_obs, obs_p0_x, gamma)

    raise NotImplementedError
