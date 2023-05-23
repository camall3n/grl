import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass, fields
from typing import Union, Iterable

@register_pytree_node_class
@dataclass
class Batch:
    obs: Union[np.ndarray, Iterable]
    action: Union[np.ndarray, Iterable]
    next_obs: Union[np.ndarray, Iterable] = None
    reward: Union[np.ndarray, Iterable] = None
    prev_action: Union[np.ndarray, Iterable] = None
    done: Union[np.ndarray, Iterable] = None
    gamma: Union[np.ndarray, Iterable] = None
    next_action: Union[np.ndarray, Iterable] = None
    state: Union[np.ndarray, Iterable] = None
    next_state: Union[np.ndarray, Iterable] = None

    # MC stuff
    returns: Union[np.ndarray, Iterable] = None

    # LSTM stuff
    zero_mask: Union[np.ndarray, Iterable] = None

    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # dict_children = {k: v for k, v in children}
        # return cls(**dict_children)
        return cls(*children)

def one_hot(x, n):
    return jnp.eye(n)[x]
