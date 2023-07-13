from argparse import Namespace
from typing import Tuple, Union

from optax import GradientTransformation
from flax import linen as nn

from grl.agent.rnn import RNNAgent
from grl.agent.multihead_rnn import MultiheadRNNAgent
from grl.mdp import POMDP, MDP

def get_agent(network: nn.Module, optimizer: GradientTransformation, features_shape: Tuple,
              env: Union[MDP, POMDP], args: Namespace):
    if args.algo == 'rnn':
        return RNNAgent(network, optimizer, features_shape, env.action_space.n, args)
    elif args.algo == 'multihead_rnn':
        return MultiheadRNNAgent(network, optimizer, features_shape, env.action_space.n, args)
    else:
        raise NotImplementedError
