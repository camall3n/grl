import numpy as np
import jax.numpy as jnp
from jax import jit, nn
from typing import Tuple
from functools import partial

from grl.utils.math import glorot_init, reverse_softmax
from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.utils import functional_solve_mdp

@partial(jit, static_argnames=['gamma'])
def value_iteration_step(vp: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, gamma: float):
    # we repeat values over S x A
    repeated_vp = vp[None, ...]
    repeated_vp = repeated_vp.repeat(R.shape[0] * R.shape[1], axis=0)
    repeated_vp_reshaped = repeated_vp.reshape(R.shape[0], R.shape[1], -1)

    # g = r + gamma * v(s')
    g = R + gamma * repeated_vp_reshaped

    # g * p(s' | s, a)
    new_v = (T * g).sum(axis=-1)

    # Take max over actions
    max_new_v = new_v.max(axis=0)
    return max_new_v

def value_iteration(T: jnp.ndarray, R: jnp.ndarray, gamma: float, tol: float = 1e-10)\
        -> jnp.ndarray:
    """
    Value iteration.
    :param T: A x S x S
    :param R: A x S x S
    :param gamma: discount rate
    :param tol: tolerance for error
    :return Value function for optimal policy.
    """
    v = jnp.zeros(T.shape[-1])
    iterations = 0

    while True:
        # value iteration step
        new_v = value_iteration_step(v, T, R, gamma)

        deltas = jnp.abs(v - new_v)

        v = new_v

        delta = deltas.max()

        iterations += 1

        if delta < tol:
            break

    return v

# def get_pi_given_v(v: jnp.ndarray):
#     """
#     Get our policy over
#     """

def td_pe(pi_phi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray,
          gamma: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    pi_ground = phi @ pi_phi
    occupancy = functional_get_occupancy(pi_ground, T, p0, gamma)

    p_pi_of_s_given_o = get_p_s_given_o(phi, occupancy)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, phi, T, R)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_phi, T_obs_obs, R_obs_obs, gamma)
    return td_v_vals, td_q_vals

def get_eps_greedy_pi(q_vals: jnp.ndarray, eps: float = 0.1) -> jnp.ndarray:
    """
    :params q_vals: |A| x |O|
    :params eps: epsilon, for epsilon-greedy policy
    """
    new_phi_pi = jnp.zeros((q_vals.shape[1], q_vals.shape[0])) + (eps / q_vals.shape[0])
    max_a = q_vals.argmax(axis=0)
    new_phi_pi = new_phi_pi.at[jnp.arange(new_phi_pi.shape[0]), max_a].add(1 - eps)
    return new_phi_pi

def policy_iteration_step(pi_params: jnp.ndarray,
                          T: jnp.ndarray,
                          R: jnp.ndarray,
                          phi: jnp.ndarray,
                          p0: jnp.ndarray,
                          gamma: float,
                          eps: float = 0.1):
    pi_phi = nn.softmax(pi_params, axis=-1)
    # now we calculate our TD model and values
    td_v_vals, td_q_vals = td_pe(pi_phi, T, R, phi, p0, gamma)

    # greedification step
    new_pi_phi = get_eps_greedy_pi(td_q_vals, eps)

    # now we need to reparameterize our policy to softmaxable pi_params
    new_pi_params = reverse_softmax(new_pi_phi)

    return new_pi_params, td_v_vals, td_q_vals

def po_policy_iteration(T: jnp.ndarray,
                        R: jnp.ndarray,
                        phi: jnp.ndarray,
                        gamma: float,
                        p0: jnp.ndarray,
                        eps: float = 0.1) -> jnp.ndarray:
    """
    Value iteration over observations.
    :param T: A x S x S
    :param R: A x S x S
    :param phi:
    :param gamma: discount rate
    :param tol: tolerance for error
    :return Value function for optimal policy.
    """
    # TODO: epsilon scheduler?
    jitted_policy_iteration_step = jit(policy_iteration_step, static_argnames=['gamma', 'eps'])

    # first initialize our random policy |O| x |A|
    pi_phi = nn.softmax(glorot_init((phi.shape[-1], T.shape[0])), axis=-1)

    # Now we
    iterations = 0

    while True:
        iterations += 1

        # now we calculate our TD model and values
        # along with a greedification step
        new_pi_phi, td_v_vals, td_q_vals = jitted_policy_iteration_step(pi_phi,
                                                                        T,
                                                                        R,
                                                                        phi,
                                                                        p0,
                                                                        gamma,
                                                                        eps=eps)

        if np.allclose(pi_phi, new_pi_phi):
            break

        pi_phi = new_pi_phi

    return pi_phi
