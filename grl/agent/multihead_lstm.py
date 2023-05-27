from argparse import Namespace
from typing import Tuple

import jax.numpy as jnp
import optax

from grl.agent.lstm import LSTMAgent
from grl.model.multihead_lstm import TwoHeadedLSTMQNetwork, LSTMCarry
from grl.utils.data import Batch

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
        hiddens, q_td, q_mc = self.network.apply(combined_params, batch.obs, batch.state)

    def _split_loss(self, lstm_params: dict, value_head_params: dict, batch: Batch):
        pass

    def update(self,
               network_params: dict,
               optimizer_state: optax.Params,
               batch: Batch):
        pass
