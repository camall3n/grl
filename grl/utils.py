import numpy as np
import jax.numpy as jnp
import pickle
from pathlib import Path
from time import time, ctime
from typing import Tuple

from pprint import pformat
from typing import Sequence

RTOL = 1e-3

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)

def golrot_init(shape: Sequence[int]) -> jnp.ndarray:
    return np.random.normal(size=shape) * np.sqrt(2)
