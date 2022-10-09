import numpy as np
import jax.numpy as jnp

from jax.nn import softmax
from pprint import pformat

RTOL = 1e-3

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)
