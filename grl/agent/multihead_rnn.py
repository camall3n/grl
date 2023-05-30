from argparse import Namespace
from typing import Tuple

from flax.linen import FrozenDict
import jax
import jax.numpy as jnp
import optax

from grl.agent.rnn import RNNAgent
from grl.model.rnn import TwoHeadedRNNQNetwork
from grl.utils.data import Batch
from grl.utils.loss import seq_sarsa_loss, seq_sarsa_mc_loss, seq_sarsa_lambda_discrep, mse

class MultiheadRNNAgent(RNNAgent):
    def __init__(self,
                 network: TwoHeadedRNNQNetwork,
                 optimizer: optax.GradientTransformation,
                 features_shape: Tuple,
                 n_actions: int,
                 args: Namespace):
        super().__init__(network, optimizer, features_shape, n_actions, args)
        self.action_mode = args.multihead_action_mode
        self.loss_mode = args.multihead_loss_mode
        self.lambda_coeff = args.multihead_lambda_coeff

    def Qs(self, network_params: dict, obs: jnp.ndarray, hidden_state: jnp.ndarray, *args,
           mode: str = None) -> jnp.ndarray:
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
            return td_q
        elif mode == 'mc':
            return mc_q
        else:
            raise NotImplementedError

    def _combined_loss(self, combined_params: dict, batch: Batch,
                       loss_mode: str = 'both', lambda_coeff: float = 0.):
        final_hidden, all_qs_td, all_qs_mc = self.network.apply(combined_params, batch.obs, batch.state)

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
            loss = batch_td_loss(q_td, actions, batch.reward, batch.gamma, q1_td, next_actions) + \
                            batch_mc_loss(q_mc, actions, batch.returns) + \
                            lambda_coeff * batch_lambda_discrep(q_td, q_mc, actions)
        else:
            assert NotImplementedError

        # Don't learn from the values past dones.
        loss *= batch.zero_mask
        return mse(loss)

    def _split_loss(self, rnn_params: dict, value_heads_params: dict, batch: Batch,
                    loss_mode: str = 'discrep'):
        """
        :param loss_mode:
        """
        all_params = FrozenDict({'params': {'rnn': rnn_params, 'value': value_heads_params}})

        pass

    def update(self,
               network_params: dict,
               optimizer_state: optax.Params,
               batch: Batch):

        if self.loss_mode in ['both', 'td', 'mc']:
            return super().update(network_params, optimizer_state, batch)
        elif self.loss_mode == 'split':
            """
            Split mode trains the RNN with ONLY lambda discrepancy, 
            and the value head with ONLY td/mc.
            """
            pass
        else:
            raise NotImplementedError

