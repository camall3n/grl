import jax.numpy as jnp
from functools import partial
from jax import jit

from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model

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

    return state_vals, mc_vals, td_vals, {'occupancy': occupancy}

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

@jit
def functional_solve_amdp(mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                          pi_abs: jnp.ndarray):
    # Q vals
    amdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

    # V vals
    amdp_v_vals = (amdp_q_vals * pi_abs.T).sum(0)

    return {'v': amdp_v_vals, 'q': amdp_q_vals}

