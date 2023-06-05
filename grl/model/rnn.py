from flax import linen as nn
import jax.numpy as jnp

def get_rnn_cell(arch: str):
    if arch == 'gru':
        return nn.GRUCell
    elif arch == 'lstm':
        return nn.OptimizedLSTMCell
    elif arch == 'elman':
        return nn.RNNCellBase
    else:
        raise NotImplementedError

class RNNQNetwork(nn.Module):
    hidden_size: int
    n_actions: int
    value_head_layers: int = 0
    arch: str = 'rnn'
    obs_value_inputs: bool = False

    def setup(self):
        self.rnn_cell_class = get_rnn_cell(self.arch)

        # define our value head MLP
        value_layers = []
        for layer_idx in range(self.value_head_layers):
            value_layers.append(nn.Dense(features=self.hidden_size))
            value_layers.append(nn.relu)

        value_layers.append(nn.Dense(features=self.n_actions))
        self.value_layers = nn.Sequential(value_layers, name='value')

    @nn.compact
    def __call__(self, obses: jnp.ndarray, init_carry: jnp.ndarray):
        """
        Applies a GRU over the init_carry for all observations in obses.
        init_carry: A tuple of jnp.ndarrays, of size b x hidden_size
        obses: observations of size b x t x *obs_size
        """
        rnn_func = nn.RNN(self.rnn_cell_class(name='rnn'), self.hidden_size, return_carry=True)

        carry, rnn_out = rnn_func(obses, initial_carry=init_carry)
        if self.obs_value_inputs:
            rnn_out = jnp.concatenate((rnn_out, obses), axis=-1)
        qs = self.value_layers(rnn_out)

        return carry, qs

class TwoHeadedRNNQNetwork(RNNQNetwork):
    def setup(self):
        self.rnn_cell_class = get_rnn_cell(self.arch)

        # define our value head MLP
        value_layers = []
        for layer_idx in range(self.value_head_layers):
            value_layers.append(nn.Dense(features=self.hidden_size))
            value_layers.append(nn.relu)

        value_layers.append(nn.Dense(features=self.n_actions * 2))
        self.value_layers = nn.Sequential(value_layers, name='value')

    @nn.compact
    def __call__(self, obses: jnp.ndarray, init_carry: jnp.ndarray):
        """
        Applies a GRU over the init_carry for all observations in obses.
        init_carry: A tuple of jnp.ndarrays, of size b x hidden_size
        obses: observations of size b x t x *obs_size
        """
        rnn_func = nn.RNN(self.rnn_cell_class(name='rnn'), self.hidden_size, return_carry=True)

        carry, rnn_out = rnn_func(obses, initial_carry=init_carry)
        concat_qs = self.value_layers(rnn_out)

        q0 = concat_qs[:, :, :self.n_actions]
        q1 = concat_qs[:, :, self.n_actions:]

        return carry, q0, q1
