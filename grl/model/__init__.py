from argparse import Namespace

from .gru import GRUQNetwork
from .multihead_gru import TwoHeadedGRUQNetwork

def get_network(args: Namespace, n_outs: int):
    if args.arch == 'gru':
        if args.algo == 'rnn':
            return GRUQNetwork(n_actions=n_outs, hidden_size=args.hidden_size)
        elif args.algo == 'multihead_rnn':
            return TwoHeadedGRUQNetwork(n_actions=n_outs, hidden_size=args.hidden_size)
        else:
            raise NotImplementedError
    elif args.arch == 'lstm':
        raise NotImplementedError
    elif args.arch == 'elman':
        raise NotImplementedError
    else:
        raise NotImplementedError

