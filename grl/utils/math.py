import numpy as np
import jax.numpy as jnp

from pprint import pformat
from typing import Sequence, Union

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)

def glorot_init(shape: Sequence[int], scale: float = 0.5) -> jnp.ndarray:
    return np.random.normal(size=shape) * scale

def normalize(arr: np.ndarray, axis=-1) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        normalized_arr = arr / np.expand_dims(arr.sum(axis), axis)
    normalized_arr = np.nan_to_num(normalized_arr)
    return normalized_arr

def greedify(pi: np.ndarray) -> np.ndarray:
    """
    pi: |O| x |A| array.
    This function returns an array of the same size, except with the max values of each row
    == 1, where everything else is 0.
    """
    pi_greedy = np.zeros_like(pi)
    pi_greedy[np.arange(pi.shape[0]), pi.argmax(axis=-1)] = 1
    return pi_greedy

def reverse_softmax(dists: jnp.ndarray, eps: float = 1e-20) -> jnp.ndarray:
    """
    A fast and efficient way to turn a distribution
    into softmax-able parameters, where
    dist = softmax(reverse_softmax(dist))
    :param dists: distribution to un-softmax. We assume that the last dimension sums to 1.
    """
    # c = jnp.log(jnp.exp(dists).sum(axis=-1))
    # params = jnp.log(dists) + c
    params = jnp.log(dists + eps)
    return params
