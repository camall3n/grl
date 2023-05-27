from argparse import Namespace

from .lstm import LSTMQNetwork
from .multihead_lstm import TwoHeadedLSTMQNetwork

def get_network(args: Namespace, n_outs: int):
    if args.algo == 'lstm':
        return LSTMQNetwork(n_actions=n_outs, hidden_size=args.hidden_size)
    elif args.algo == 'multihead_lstm':
        return TwoHeadedLSTMQNetwork(n_actions=n_outs, hidden_size=args.hidden_size)
    else:
        raise NotImplementedError

