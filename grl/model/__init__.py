from argparse import Namespace

from .lstm import LSTMQNetwork

def get_network(args: Namespace, n_outs: int):
    if args.algo == 'lstm':
        return LSTMQNetwork(n_actions=n_outs, hidden_size=args.hidden_size)
    else:
        raise NotImplementedError

