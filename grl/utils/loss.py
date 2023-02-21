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
                        value_type: str = 'q', error_type: str = 'l2', alpha: float = 1.,
                        flip_count_prob: bool = False):  # initialize with partial
    pi = nn.softmax(pi_params, axis=-1)
    loss, mc_vals, td_vals = discrep_loss(pi, amdp, value_type, error_type, alpha, flip_count_prob=flip_count_prob)
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


def mem_magnitude_td_loss(mem_params: jnp.ndarray, pi: jnp.ndarray, amdp: AbstractMDP,  # input non-static arrays
                          value_type: str = 'q', error_type: str = 'l2', alpha: float = 1.,
                          flip_count_prob: bool = False):
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    loss, _, _ = magnitude_td_loss(pi, mem_aug_amdp, value_type, error_type, alpha, flip_count_prob=flip_count_prob)
    return loss

@partial(jit, static_argnames=['value_type', 'error_type', 'alpha', 'flip_count_prob'])
def magnitude_td_loss(pi: jnp.ndarray, amdp: AbstractMDP,  # non-state args
                 value_type: str = 'q', error_type: str = 'l2', alpha: float = 1.,
                 flip_count_prob: bool = False): # initialize static args
    # TODO: this is wrong?
    _, mc_vals, td_vals, info = analytical_pe(pi, amdp)
    assert value_type == 'q'
    R_s_o = amdp.R @ amdp.phi  # A x S x O
    expanded_R_s_o = jnp.expand_dims(R_s_o, -1).repeat(amdp.n_actions, axis=-1)


    # repeat the Q-function over A x O
    # Multiply that with p(O', A' | s, a) and sum over O' and A' dimensions.
    # P(O' | s, a) = T @ phi, P(A', O' | s, a) = P(O' | s, a) * pi (over new dimension)
    pr_o = amdp.T @ amdp.phi
    pr_o_a = jnp.einsum('ijk,kl->ijkl', pr_o, pi)
    expected_next_Q = jnp.einsum('ijkl,kl->ijkl', pr_o_a, td_vals['q'].T)
    expanded_Q = jnp.expand_dims(jnp.expand_dims(td_vals['q'].T, 0).repeat(pr_o_a.shape[1], axis=0), 0).repeat(pr_o_a.shape[0], axis=0)
    diff = expanded_R_s_o + amdp.gamma * expected_next_Q - expanded_Q

    c_s = info['occupancy']
    # set terminal counts to 0
    c_s = c_s.at[-2:].set(0)
    count_s = c_s / c_s.sum()

    if flip_count_prob:
        count_s = nn.softmax(-count_s)

    uniform_s = jnp.ones(amdp.n_states) / amdp.n_states

    p_s = alpha * uniform_s + (1 - alpha) * count_s

    pi_states = amdp.phi @ pi
    weight = (pi_states * p_s[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    expanded_weight = jnp.expand_dims(jnp.expand_dims(weight, -1).repeat(pr_o_a.shape[2], axis=-1), -1).repeat(pr_o_a.shape[-1], axis=-1)
    weighted_err = expanded_weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    loss = weighted_err.sum()

    return loss, mc_vals, td_vals
