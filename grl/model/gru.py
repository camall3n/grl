from typing import Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

class GRUQNetwork(nn.Module):
    hidden_size: int
    n_actions: int

    @nn.compact
    def __call__(self, obses: jnp.ndarray, init_carry: jnp.ndarray):
        """
        Applies a GRU over the init_carry for all observatiosn in obses.
        init_carry: A tuple of jnp.ndarrays, of size b x hidden_size
        obses: observations of size b x t x *obs_size
        """
        gru_func = nn.RNN(nn.GRUCell(name='rnn'), self.hidden_size, return_carry=True)
        val_head = nn.Dense(features=self.n_actions, name='value')

        carry, gru_out = gru_func(obses, initial_carry=init_carry)
        qs = val_head(gru_out)

        return carry, qs
