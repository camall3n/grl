from typing import Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

LSTMCarry = Tuple[jax.Array, jax.Array]

class LSTMQNetwork(nn.Module):
    hidden_size: int
    n_actions: int

    @nn.compact
    def __call__(self, obses: jnp.ndarray, init_carry: LSTMCarry):
        """
        Applies an LSTM over the init_carry for all observatiosn in obses.
        init_carry: A tuple of jnp.ndarrays, of size b x hidden_size
        obses: observations of size b x t x *obs_size
        """
        lstm_func = nn.RNN(nn.LSTMCell(name='lstm'), self.hidden_size, return_carry=True)
        val_head = nn.Dense(features=self.n_actions, name='value')

        carry, lstm_out = lstm_func(obses, initial_carry=init_carry)
        qs = val_head(lstm_out)

        return carry, qs
