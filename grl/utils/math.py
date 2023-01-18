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

def softmax(arr: np.ndarray) -> np.ndarray:
    assert len(arr.shape) == 2
    exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
