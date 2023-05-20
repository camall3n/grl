from argparse import Namespace
from typing import Tuple

from flax import linen as nn
import jax
from jax import random
from optax import GradientTransformation
import jax.numpy as jnp

LSTMCarry = Tuple[jax.Array, jax.Array]

class LSTMAgent:
    def __init__(self,
                 network: nn.Module,
                 optimizer: GradientTransformation,
                 rand_key: random.PRNGKey,
                 features_shape: Tuple,
                 n_actions: int,
                 args: Namespace):
        self.features_shape = features_shape
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        self.trunc = args.trunc
        self.init_hidden_var = args.init_hidden_var
        self.hidden_state = None

        self._rand_key, network_rand_key = random.split(rand_key)
        self.network = network
        self.reset()
        self.network_params = self.network.init(rng=network_rand_key,
                                                x=jnp.zeros((1, self.trunc, *self.features_shape)),
                                                h=self.hidden_state, train=True)
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon
        self.args = args
        self.curr_q = None

        self.er_hidden_update = args.er_hidden_update

    def reset(self):



