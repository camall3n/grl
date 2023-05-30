from argparse import Namespace
from typing import Tuple

from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
import optax
from optax import GradientTransformation

from grl.model.rnn import rnn_arch_module_switch
from grl.utils.data import Batch
from grl.utils.loss import mse, seq_sarsa_loss

class RNNAgent:
    def __init__(self,
                 network: nn.Module,
                 optimizer: GradientTransformation,
                 features_shape: Tuple,
                 n_actions: int,
                 args: Namespace):

        self.features_shape = features_shape
        self.n_hidden = args.hidden_size
        self.n_actions = n_actions

        self.trunc = args.trunc
        self.eps = args.epsilon
        self.args = args

        self.network = network
        self.optimizer = optimizer

        self.act = jax.jit(self.act)
        self.update = jax.jit(self.update)

    def init_params(self, rand_key: random.PRNGKey) -> Tuple[dict, optax.Params, random.PRNGKey]:
        """
        Initialize params for agent and optimizer here.
        """
        rand_key, network_key = random.split(rand_key)
        new_carry, rand_key = self.reset(rand_key)
        network_params = self.network.init(network_key,
                                           jnp.zeros((1, 1, *self.features_shape)),
                                           new_carry)

        optimizer_params = self.optimizer.init(network_params)
        return network_params, optimizer_params, rand_key

    def reset(self, rand_key: random.PRNGKey) -> Tuple[jnp.ndarray, random.PRNGKey]:
        """
        Reset the RNN initial state.
        """
        rand_key, carry_key = random.split(rand_key)
        rnn_module_class = rnn_arch_module_switch(self.args.arch)
        new_carry = rnn_module_class.initialize_carry(carry_key, (1, ), self.n_hidden)
        return new_carry, rand_key

    def act(self, network_params: dict, obs: jnp.ndarray, hs: jnp.ndarray, rand_key: random.PRNGKey):
        """
        Given an observation, act based on self.hidden_state.
        obs should be of size n_obs, with NO batch dimension.
        """
        obs = jnp.expand_dims(jnp.expand_dims(obs, 0), 0)  # bs x ts x *obs_size, bs = ts = 1
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, new_hidden_state, qs = self.greedy_act(network_params, obs, hs)
        probs = probs.at[greedy_idx].add(1 - self.eps)

        key, subkey = random.split(rand_key)
        selected_action = random.choice(subkey, jnp.arange(self.n_actions),
                                        p=probs, shape=(obs.shape[0],))

        return selected_action, key, new_hidden_state, qs

    def greedy_act(self, network_params: dict, obs: jnp.ndarray, hidden_state: jnp.ndarray) -> \
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given an obs and hidden_state
        :param network_params: Optional. Potentially use another model to find action-values.
        :param obs: (b x 1 x *obs.shape) Obs to find actions
        :param hidden_state: RNN hidden state
        :return: (b) Greedy actions
        """
        new_hidden_state, qs = self.Qs(network_params, obs, hidden_state)
        return jnp.argmax(qs[:, 0], axis=1), new_hidden_state, qs

    def Qs(self, network_params: dict, obs: jnp.ndarray, hidden_state: jnp.ndarray, *args) -> jnp.ndarray:
        """
        Get all Q-values given an observation and hidden state.
        :param network_params: network params to find Qs w.r.t.
        :param obs: (b x t x *obs_size) Obs to find action-values
        :param hidden_state: RNN hidden state to propagate.
        :return: (b x t x actions)  full of action-values.
        """
        return self.network.apply(network_params, obs, hidden_state)

    def _loss(self, network_params: dict, batch: Batch):
        final_hidden, all_qs = self.network.apply(network_params, batch.obs, batch.state)

        q, q1 = all_qs[:, :-1], all_qs[:, 1:]
        actions, next_actions = batch.action[:, :-1], batch.action[:, 1:]

        batch_loss = jax.vmap(seq_sarsa_loss)
        td_err = batch_loss(q, actions, batch.reward, batch.gamma, q1, next_actions)  # Should be batch x seq_len

        # Don't learn from the values past dones.
        td_err *= batch.zero_mask
        return mse(td_err)

    def update(self,
               network_params: dict,
               optimizer_state: optax.Params,
               batch: Batch) -> \
        Tuple[dict, dict, optax.Params]:
        """
        :return: loss, network parameters, optimizer state and all hidden states (bs x timesteps x 2 x n_hidden)
        """
        # We only take t=0.
        if isinstance(batch.state, tuple):
            batch.state = tuple(state[:, 0] for state in batch.state)
        else:
            batch.state = batch.state[:, 0]

        loss, grad = jax.value_and_grad(self._loss)(network_params, batch)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return network_params, optimizer_state, {'total_loss': loss}
