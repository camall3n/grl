import jax.numpy as jnp
from jax import nn, lax, jit
from functools import partial

from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.utils.policy_eval import analytical_pe, functional_solve_mdp
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
"""
The following few functions are loss function w.r.t. memory parameters, mem_params.
"""

@partial(jit, static_argnames=['value_type', 'error_type', 'alpha', 'flip_count_prob'])
def discrep_loss(pi: jnp.ndarray, amdp: AbstractMDP,  # non-state args
                 value_type: str = 'q', error_type: str = 'l2', alpha: float = 1.,
                 flip_count_prob: bool = False): # initialize static args
    _, mc_vals, td_vals, info = analytical_pe(pi, amdp)
    diff = mc_vals[value_type] - td_vals[value_type]

    c_s = info['occupancy']
    # set terminal counts to 0
    c_s = c_s.at[-2:].set(0)
    c_o = c_s @ amdp.phi
    count_o = c_o / c_o.sum()

    if flip_count_prob:
        count_o = nn.softmax(-count_o)

    uniform_o = jnp.ones(pi.shape[0]) / pi.shape[0]

    p_o = alpha * uniform_o + (1 - alpha) * count_o

    weight = (pi * p_o[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    loss = weighted_err.sum()

    return loss, mc_vals, td_vals

def mem_discrep_loss(mem_params: jnp.ndarray, pi: jnp.ndarray, amdp: AbstractMDP,  # input non-static arrays
                     value_type: str = 'q', error_type: str = 'l2', alpha: float = 1.,
                     flip_count_prob: bool = False):  # initialize with partial
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    loss, _, _ = discrep_loss(pi, mem_aug_amdp, value_type, error_type, alpha, flip_count_prob=flip_count_prob)
    return loss

def policy_discrep_loss(pi_params: jnp.ndarray, amdp: AbstractMDP,
                        value_type: str = 'q', error_type: str = 'l2', alpha: float = 1.):
    pi = nn.softmax(pi_params, axis=-1)
    loss, mc_vals, td_vals = discrep_loss(pi, amdp, value_type, error_type, alpha)
    return loss, (mc_vals, td_vals)

def pg_objective_func(pi_params: jnp.ndarray, amdp: AbstractMDP):
    """
    Policy gradient objective function:
    sum_{s_0} p(s_0) v_pi(s_0)
    """
    pi_abs = nn.softmax(pi_params, axis=-1)
    pi_ground = amdp.phi @ pi_abs
    occupancy = functional_get_occupancy(pi_ground, amdp)

    p_pi_of_s_given_o = get_p_s_given_o(amdp.phi, occupancy)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, amdp)
    td_model = MDP(T_obs_obs, R_obs_obs, amdp.p0 @ amdp.phi, gamma=amdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_abs, td_model)
    p_init_obs = amdp.p0 @ amdp.phi
    return jnp.dot(p_init_obs, td_v_vals), (td_v_vals, td_q_vals)

def mem_abs_td_loss(mem_params: jnp.ndarray, pi: jnp.ndarray, amdp: AbstractMDP,  # input non-static arrays
                    value_type: str, error_type: str, weight_discrep_by_count: bool):  # initialize with partial
    """
    Absolute TD error loss.
    This is an upper bound on absolute lambda discrepancy.
    """
    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    _, mc_vals, td_vals, info = analytical_pe(pi, mem_aug_amdp)
    vals = td_vals[value_type]

    weight = pi.T if value_type == 'q' else jnp.ones_like(vals)
    if weight_discrep_by_count:
        c_s = info['occupancy']
        c_o = c_s @ amdp.phi
        p_o = c_o / c_o.sum()
        weight = lax.stop_gradient(pi * p_o[:, None]).T
        if value_type == 'v':
            weight = weight.sum(axis=0)

    if error_type == 'l2':
        unweighted_err = (vals**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(vals)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = unweighted_err.sum(axis=0)

    if weight_discrep_by_count:
        loss = weighted_err.sum()
    else:
        loss = weighted_err.mean()

    return loss, mc_vals, td_vals
