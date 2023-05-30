from argparse import Namespace
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from grl.agent.lstm import LSTMAgent
from grl.model.multihead_lstm import TwoHeadedLSTMQNetwork, LSTMCarry
from grl.utils.data import Batch
from grl.utils.loss import seq_sarsa_loss, seq_sarsa_mc_loss, seq_sarsa_lambda_discrep, mse

class MultiheadLSTMAgent(LSTMAgent):
    def __init__(self,
                 network: TwoHeadedLSTMQNetwork,
                 optimizer: optax.GradientTransformation,
                 features_shape: Tuple,
                 n_actions: int,
                 args: Namespace):
        super().__init__(network, optimizer, features_shape, n_actions, args)
        self.action_mode = args.multihead_action_mode
        self.loss_mode = args.multihead_loss_mode
        self.lambda_coeff = args.multihead_lambda_coeff

    def Qs(self, network_params: dict, obs: jnp.ndarray, hidden_state: LSTMCarry, *args,
           mode: str = None) -> jnp.ndarray:
        """
        Get all Q-values given an observation and hidden state, from the self.action_mode head.
        :param network_params: network params to find Qs w.r.t.
        :param obs: (b x t x *obs_size) Obs to find action-values
        :param hidden_state: LSTM hidden state to propagate.
        :return: (b x t x actions)  full of action-values.
        """
        if mode is None:
            mode = self.action_mode

        carry, td_q, mc_q = self.network.apply(network_params, obs, hidden_state)

        if mode == 'td':
            return td_q
        elif mode == 'mc':
            return mc_q
        else:
            raise NotImplementedError

    def _combined_loss(self, combined_params: dict, batch: Batch):
        final_hidden, all_qs_td, all_qs_mc = self.network.apply(combined_params, batch.obs, batch.state)

        q_td, q1_td = all_qs_td[:, :-1], all_qs_td[:, 1:]
        q_mc, q1_mc = all_qs_mc[:, :-1], all_qs_mc[:, 1:]
        actions, next_actions = batch.action[:, :-1], batch.action[:, 1:]

        batch_td_loss = jax.vmap(seq_sarsa_loss)
        batch_mc_loss = jax.vmap(seq_sarsa_mc_loss)
        batch_lambda_discrep = jax.vmap(seq_sarsa_lambda_discrep)

        if self.loss_mode == 'td':
            loss = batch_td_loss(q_td, actions, batch.reward, batch.gamma, q1_td, next_actions)
        elif self.loss_mode == 'mc':
            loss = batch_mc_loss(q_mc, actions, batch.returns)
        elif self.loss_mode == 'both':
            loss = batch_td_loss(q_td, actions, batch.reward, batch.gamma, q1_td, next_actions) + \
                            batch_mc_loss(q_mc, actions, batch.returns) + \
                            self.lambda_coeff * batch_lambda_discrep(q_td, q_mc, actions)
        else:
            assert NotImplementedError

        # Don't learn from the values past dones.
        loss *= batch.zero_mask
        return mse(loss)

    def _split_loss(self, lstm_params: dict, value_head_params: dict, batch: Batch):
        pass

    def update(self,
               network_params: dict,
               optimizer_state: optax.Params,
               batch: Batch):
        pass
