from argparse import Namespace
from typing import Tuple

from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
import optax
from optax import GradientTransformation

from grl.model.lstm import LSTMCarry
from grl.utils.data import Batch
from grl.utils.loss import mse, seq_sarsa_loss


class LSTMAgent:
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

    def reset(self, rand_key: random.PRNGKey) -> Tuple[LSTMCarry, random.PRNGKey]:
        """
        Reset the LSTM initial state (called LSTMCarry here).
        """
        rand_key, carry_key = random.split(rand_key)
        new_carry = nn.OptimizedLSTMCell.initialize_carry(carry_key, (1, ), self.n_hidden)
        return new_carry, rand_key

    def act(self, network_params: dict, obs: jnp.ndarray, hs: LSTMCarry, rand_key: random.PRNGKey):
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

    def greedy_act(self, network_params: dict, obs: jnp.ndarray, hidden_state: LSTMCarry) -> \
            Tuple[jnp.ndarray, LSTMCarry, jnp.ndarray]:
        """
        Get greedy actions given an obs and hidden_state
        :param network_params: Optional. Potentially use another model to find action-values.
        :param obs: (b x 1 x *obs.shape) Obs to find actions
        :param hidden_state: LSTM hidden state
        :return: (b) Greedy actions
        """
        new_hidden_state, qs = self.Qs(network_params, obs, hidden_state)
        return jnp.argmax(qs[:, 0], axis=1), new_hidden_state, qs

    def Qs(self, network_params: dict, obs: jnp.ndarray, hidden_state: LSTMCarry, *args) -> jnp.ndarray:
        """
        Get all Q-values given an observation and hidden state.
        :param network_params: network params to find Qs w.r.t.
        :param obs: (b x t x *obs_size) Obs to find action-values
        :param hidden_state: LSTM hidden state to propagate.
        :return: (b x t x actions)  full of action-values.
        """
        return self.network.apply(network_params, obs, hidden_state)

    def _loss(self, network_params: dict, batch: Batch):
        q, hiddens, _ = self.network.apply(network_params, batch.obs, batch.state)
        q1, target_hiddens, _ = self.network.apply(network_params, batch.next_obs, batch.next_state)

        batch_loss = jax.vmap(seq_sarsa_loss)
        td_err = batch_loss(q, batch.action, batch.reward, batch.gamma, q1, batch.next_action)  # Should be batch x seq_len

        # Don't learn from the values past dones.
        td_err *= batch.zero_mask
        return mse(td_err), (hiddens, target_hiddens)

    def update(self,
               network_params: dict,
               optimizer_state: optax.Params,
               batch: Batch):
        """
        :return: loss, network parameters, optimizer state and all hidden states (bs x timesteps x 2 x n_hidden)
        """
        loss, grad = jax.value_and_grad(self._loss)(network_params, batch)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state
