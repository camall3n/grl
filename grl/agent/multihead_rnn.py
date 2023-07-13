from argparse import Namespace
from dataclasses import replace
from typing import Tuple, Union

from flax.linen import FrozenDict
import jax
from jax import random
import jax.numpy as jnp
import optax

from grl.agent.rnn import RNNAgent
from grl.model.rnn import TwoHeadedRNNQNetwork
from grl.utils.data import Batch
from grl.utils.loss import seq_sarsa_loss, seq_sarsa_mc_loss, seq_sarsa_lambda_discrep, mse

class MultiheadRNNAgent(RNNAgent):
    def __init__(self, network: TwoHeadedRNNQNetwork, optimizer: optax.GradientTransformation,
                 features_shape: Tuple, n_actions: int, args: Namespace):
        super().__init__(network, optimizer, features_shape, n_actions, args)
        self.action_mode = args.multihead_action_mode
        self.loss_mode = args.multihead_loss_mode
        self.lambda_coeff = args.multihead_lambda_coeff

    def init_params(self, rand_key: random.PRNGKey) -> \
            Tuple[dict, Union[optax.Params, dict], random.PRNGKey]:
        network_params, optimizer_params, rand_key = super().init_params(rand_key)
        if self.loss_mode == 'split':
            rnn_optimizer_params = self.optimizer.init(network_params['params']['rnn'])
            value_optimizer_params = self.optimizer.init(network_params['params']['value'])
            optimizer_params = {'rnn': rnn_optimizer_params, 'value': value_optimizer_params}
        return network_params, optimizer_params, rand_key

    def Qs(self,
           network_params: dict,
           obs: jnp.ndarray,
           hidden_state: jnp.ndarray,
           *args,
           mode: str = None) -> Tuple[Union[jnp.ndarray, Tuple], jnp.ndarray]:
        """
        Get all Q-values given an observation and hidden state, from the self.action_mode head.
        :param network_params: network params to find Qs w.r.t.
        :param obs: (b x t x *obs_size) Obs to find action-values
        :param hidden_state: RNN hidden state to propagate.
        :return: (b x t x actions)  full of action-values.
        """
        if mode is None:
            mode = self.action_mode

        carry, td_q, mc_q = self.network.apply(network_params, obs, hidden_state)

        if mode == 'td':
            return carry, td_q
        elif mode == 'mc':
            return carry, mc_q
        elif mode == 'both':
            return carry, td_q, mc_q
        else:
            raise NotImplementedError

    def _combined_loss(self,
                       combined_params: dict,
                       batch: Batch,
                       loss_mode: str = 'both',
                       lambda_coeff: float = 0.):
        final_hidden, all_qs_td, all_qs_mc = self.network.apply(combined_params, batch.obs,
                                                                batch.state)

        q_td, q1_td = all_qs_td[:, :-1], all_qs_td[:, 1:]
        q_mc, q1_mc = all_qs_mc[:, :-1], all_qs_mc[:, 1:]
        actions, next_actions = batch.action[:, :-1], batch.action[:, 1:]

        batch_td_loss = jax.vmap(seq_sarsa_loss)
        batch_mc_loss = jax.vmap(seq_sarsa_mc_loss)
        batch_lambda_discrep = jax.vmap(seq_sarsa_lambda_discrep)

        if loss_mode == 'td':
            loss = batch_td_loss(q_td, actions, batch.reward, batch.gamma, q1_td, next_actions)
        elif loss_mode == 'mc':
            loss = batch_mc_loss(q_mc, actions, batch.returns)
        elif loss_mode == 'both':
            td_loss = batch_td_loss(q_td, actions, batch.reward, batch.gamma, q1_td, next_actions)
            mc_loss = batch_mc_loss(q_mc, actions, batch.returns)
            discrep_loss = batch_lambda_discrep(q_td, q_mc, actions)
            loss = mse(td_loss, zero_mask=batch.zero_mask) + \
                   mse(mc_loss, zero_mask=batch.zero_mask) + \
                   lambda_coeff * mse(discrep_loss, zero_mask=batch.zero_mask)
            return loss

        elif loss_mode == 'discrep':
            loss = batch_lambda_discrep(q_td, q_mc, actions)
        else:
            assert NotImplementedError

        # Don't learn from the values past dones.
        return mse(loss, zero_mask=batch.zero_mask)

    def _split_loss(self,
                    rnn_params: dict,
                    value_heads_params: dict,
                    batch: Batch,
                    loss_mode: str = 'both',
                    lambda_coeff: float = 0):
        """
        :param loss_mode:
        """
        all_params = FrozenDict({'params': {'rnn': rnn_params, 'value': value_heads_params}})

        return self._combined_loss(all_params,
                                   batch,
                                   loss_mode=loss_mode,
                                   lambda_coeff=lambda_coeff)

    def update(self,
               network_params: dict,
               optimizer_state: Union[optax.Params, dict],
               batch: Batch) -> \
        Tuple[dict, dict, optax.Params]:

        # We only take t=0.
        if isinstance(batch.state, tuple):
            batch = replace(batch, state=tuple(state[:, 0] for state in batch.state))
        else:
            batch = replace(batch, state=batch.state[:, 0])

        if self.loss_mode in ['both', 'td', 'mc']:
            loss, grad = jax.value_and_grad(self._combined_loss)(network_params,
                                                                 batch,
                                                                 loss_mode=self.loss_mode,
                                                                 lambda_coeff=self.lambda_coeff)
            updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
            network_params = optax.apply_updates(network_params, updates)
            info = {'total_loss': loss}

        elif self.loss_mode == 'split':
            """
            Split mode trains the RNN with ONLY lambda discrepancy, 
            and the value head with ONLY td/mc.
            """
            og_rnn_params = network_params['params']['rnn']
            og_value_params = network_params['params']['value']

            # RNN update
            rnn_loss, rnn_grad = \
                jax.value_and_grad(self._split_loss, argnums=0)(og_rnn_params, og_value_params, batch,
                                                                loss_mode='both')
            rnn_updates, rnn_optimizer_state = self.optimizer.update(rnn_grad,
                                                                     optimizer_state['rnn'],
                                                                     og_rnn_params)
            rnn_params = optax.apply_updates(og_rnn_params, rnn_updates)

            # Value head update
            value_loss, value_grad = \
                jax.value_and_grad(self._split_loss, argnums=1)(og_rnn_params, og_value_params, batch,
                                                                loss_mode='discrep')
            value_updates, value_optimizer_state = \
                self.optimizer.update(value_grad, optimizer_state['value'], og_value_params)
            value_params = optax.apply_updates(og_value_params, value_updates)

            network_params = FrozenDict({'params': {'rnn': rnn_params, 'value': value_params}})
            optimizer_state = {'rnn': rnn_optimizer_state, 'value': value_optimizer_state}
            info = {
                'rnn_loss': rnn_loss,
                'value_loss': value_loss,
                'total_loss': rnn_loss + rnn_loss
            }

        else:
            raise NotImplementedError

        return network_params, optimizer_state, info
