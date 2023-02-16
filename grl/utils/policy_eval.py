import jax.numpy as jnp
from functools import partial
from typing import Union
from jax import jit

from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.mdp import MDP, AbstractMDP

@jit
def analytical_pe(pi_obs: jnp.ndarray, amdp: AbstractMDP):
    # observation policy, but expanded over states
    pi_state = amdp.phi @ pi_obs

    # MC*
    state_v, state_q = functional_solve_mdp(pi_state, amdp)
    state_vals = {'v': state_v, 'q': state_q}

    occupancy = functional_get_occupancy(pi_state, amdp)

    p_pi_of_s_given_o = get_p_s_given_o(amdp.phi, occupancy)
    mc_vals = functional_solve_amdp(state_q, p_pi_of_s_given_o, pi_obs)

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, amdp)
    td_model = MDP(T_obs_obs, R_obs_obs, amdp.p0 @ amdp.phi, gamma=amdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_obs, td_model)
    td_vals = {'v': td_v_vals, 'q': td_q_vals}

    return state_vals, mc_vals, td_vals, {'occupancy': occupancy, 'R_obs_obs': R_obs_obs, 'T_obs_obs': T_obs_obs}

@jit
def functional_solve_mdp(pi: jnp.ndarray, mdp: Union[MDP, AbstractMDP]):
    """
    Solves for V using linear equations.
    For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
    """
    Pi_pi = pi.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)
    R_pi = (Pi_pi * mdp.T * mdp.R).sum(axis=0).sum(axis=-1) # R^π(s)

    # A*V_pi(s) = b
    # A = (I - \gamma (T^π))
    # b = R^π
    A = (jnp.eye(mdp.T.shape[-1]) - mdp.gamma * T_pi)
    b = R_pi
    v_vals = jnp.linalg.solve(A, b)

    R_sa = (mdp.T * mdp.R).sum(axis=-1) # R(s,a)
    q_vals = (R_sa + (mdp.gamma * mdp.T @ v_vals))

    return v_vals, q_vals

@jit
def functional_solve_amdp(mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                          pi_abs: jnp.ndarray):
    # Q vals
    amdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

    # V vals
    amdp_v_vals = (amdp_q_vals * pi_abs.T).sum(0)

    return {'v': amdp_v_vals, 'q': amdp_q_vals}

