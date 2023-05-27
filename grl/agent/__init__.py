from argparse import Namespace
from typing import Tuple, Union

from optax import GradientTransformation
from flax import linen as nn

from grl.agent.lstm import LSTMAgent
from grl.agent.multihead_lstm import MultiheadLSTMAgent
from grl.mdp import AbstractMDP, MDP

def get_agent(network: nn.Module, optimizer: GradientTransformation,
              features_shape: Tuple, env: Union[MDP, AbstractMDP],
              args: Namespace):
    if args.algo == 'lstm':
        return LSTMAgent(network, optimizer, features_shape, env.n_actions, args)
    elif args.algo == 'multihead_lstm':
        return MultiheadLSTMAgent(network, optimizer, features_shape, env.n_actions, args)
    else:
        raise NotImplementedError
