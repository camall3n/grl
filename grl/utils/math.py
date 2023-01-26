import jax.numpy as jnp
import numpy as np
import scipy.optimize
from scipy.special import logsumexp, softmax

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

def one_hot(x, n, axis=-1):
    output = np.eye(n)[x]
    return np.moveaxis(output, -1, axis)

def arg_hardmax(x, axis=-1):
    best_a = np.argmax(x, axis=axis)
    greedy_pi = one_hot(best_a, x.shape[axis], axis=axis)
    return greedy_pi

def arg_boltzman(x, axis=-1, beta=1.):
    numer = np.exp(beta * x)
    denom = np.sum(numer, axis=axis, keepdims=True)
    return (numer / denom)

def mellowmax(x, axis=-1, beta=3.9):
    n = x.shape[axis]
    return (logsumexp(beta * x, axis=axis) - np.log(n)) / beta

def arg_mellowmax(x, axis=-1, beta=3.9, beta_min=-10, beta_max=10):
    axis_last = np.moveaxis(x, axis, -1)
    mm = mellowmax(axis_last, beta=beta, axis=-1)
    batch_adv = axis_last - np.broadcast_to(np.expand_dims(mm, -1), axis_last.shape)
    batch_beta = np.empty(mm.shape, dtype=np.float32)

    # Beta is computed as the root of this function
    def f(y, adv):
        return np.sum(np.exp(y * adv) * adv)

    for idx in np.ndindex(mm.shape):
        idx_full = idx + (slice(None), )
        adv = batch_adv[idx_full]
        try:
            beta = scipy.optimize.brentq(f, a=beta_min, b=beta_max, args=(adv, ))
        except ValueError:
            beta = 0
        batch_beta[idx] = beta

    softmax_last = softmax(np.expand_dims(np.asarray(batch_beta), -1) * axis_last, axis=-1)
    return np.moveaxis(softmax_last, -1, axis)
