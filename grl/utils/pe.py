from jax import jit, nn
import jax.numpy as jnp
from functools import partial

from grl.memory import functional_memory_cross_product

@partial(jit, static_argnames=['gamma'])
def functional_get_occupancy(pi_ground: jnp.ndarray, T: jnp.ndarray, p0: jnp.ndarray,
                             gamma: float):
    Pi_pi = pi_ground.transpose()[..., None]
    T_pi = (Pi_pi * T).sum(axis=0) # T^π(s'|s)

    # A*C_pi(s) = b
    # A = (I - \gamma (T^π)^T)
    # b = P_0
    A = jnp.eye(T.shape[-1]) - gamma * T_pi.transpose()
    b = p0
    return jnp.linalg.solve(A, b)

@jit
def get_p_s_given_o(phi: jnp.ndarray, occupancy: jnp.ndarray):
    repeat_occupancy = jnp.repeat(occupancy[..., None], phi.shape[-1], -1)

    # Q vals
    p_of_o_given_s = phi.astype(float)
    w = repeat_occupancy * p_of_o_given_s

    p_pi_of_s_given_o = w / (w.sum(axis=0) + 1e-10)
    return p_pi_of_s_given_o

@partial(jit, static_argnames=['gamma'])
def functional_solve_mdp(pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, gamma: float):
    """
    Solves for V using linear equations.
    For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
    """
    Pi_pi = pi.transpose()[..., None]
    T_pi = (Pi_pi * T).sum(axis=0) # T^π(s'|s)
    R_pi = (Pi_pi * T * R).sum(axis=0).sum(axis=-1) # R^π(s)

    # A*V_pi(s) = b
    # A = (I - \gamma (T^π))
    # b = R^π
    A = (jnp.eye(T.shape[-1]) - gamma * T_pi)
    b = R_pi
    v_vals = jnp.linalg.solve(A, b)

    R_sa = (T * R).sum(axis=-1) # R(s,a)
    q_vals = (R_sa + (gamma * T @ v_vals))

    return v_vals, q_vals

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


@jit
def functional_solve_amdp(mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                          pi_abs: jnp.ndarray):
    # Q vals
    amdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

    # V vals
    amdp_v_vals = (amdp_q_vals * pi_abs.T).sum(0)

    return {'v': amdp_v_vals, 'q': amdp_q_vals}

@jit
def functional_create_td_model(p_pi_of_s_given_o: jnp.ndarray, phi: jnp.ndarray, T: jnp.ndarray,
                               R: jnp.ndarray):
    # creates an (n_obs * n_obs) x 2 array of all possible observation to observation pairs.
    # we flip here so that we have curr_obs, next_obs (order matters).
    obs_idx_product = jnp.flip(
        jnp.dstack(jnp.meshgrid(jnp.arange(phi.shape[-1]),
                                jnp.arange(phi.shape[-1]))).reshape(-1, 2), -1)

    # this gives us (n_obs * n_obs) x states x 1 and (n_obs * n_obs) x 1 x states
    curr_s_given_o = p_pi_of_s_given_o[:, obs_idx_product[:, 0]].T[..., None]
    next_o_given_s = jnp.expand_dims(phi[:, obs_idx_product[:, 1]].T, 1)

    # outer product here
    o_to_next_o = jnp.expand_dims(curr_s_given_o * next_o_given_s, 1)

    # This is p(o, s, a, s', o')
    # the probability that o goes to o', via each path (s, a) -> s'.
    # Shape is (n_obs * n_obs) x |A| x |S| x |S|
    T_contributions = T * o_to_next_o

    # |A| x (n_obs * n_obs)
    T_obs_obs_flat = T_contributions.sum(-1).sum(-1).T

    # |A| x n_obs x n_obs
    T_obs_obs = T_obs_obs_flat.reshape(T.shape[0], phi.shape[-1], phi.shape[-1])

    # You want everything to sum to one
    denom = T_obs_obs_flat.T[..., None, None]
    denom_no_zero = denom + (denom == 0).astype(denom.dtype)

    R_contributions = (R * T_contributions) / denom_no_zero
    R_obs_obs_flat = R_contributions.sum(-1).sum(-1).T
    R_obs_obs = R_obs_obs_flat.reshape(R.shape[0], phi.shape[-1], phi.shape[-1])

    return T_obs_obs, R_obs_obs

@partial(jit, static_argnames=['gamma'])
def analytical_pe(pi_obs: jnp.ndarray, phi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray,
                  p0: jnp.ndarray, gamma: float):
    # observation policy, but expanded over states
    pi_state = phi @ pi_obs

    # MC*
    state_v, state_q = functional_solve_mdp(pi_state, T, R, gamma)
    state_vals = {'v': state_v, 'q': state_q}

    occupancy = functional_get_occupancy(pi_state, T, p0, gamma)

    p_pi_of_s_given_o = get_p_s_given_o(phi, occupancy)
    mc_vals = functional_solve_amdp(state_q, p_pi_of_s_given_o, pi_obs)

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, phi, T, R)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_obs, T_obs_obs, R_obs_obs, gamma)
    td_vals = {'v': td_v_vals, 'q': td_q_vals}

    return state_vals, mc_vals, td_vals
