import numpy as np
import jax.numpy as jnp
from jax import jit

from functools import partial

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

def po_value_iteration(T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                       gamma: float, tol: float = 1e-10) -> jnp.ndarray:
    """
    Value iteration over observations.
    :param T: A x S x S
    :param R: A x S x S
    :param phi:
    :param gamma: discount rate
    :param tol: tolerance for error
    :return Value function for optimal policy.
    """

    v = jnp.zeros(phi.shape[-1])
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
