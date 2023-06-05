from argparse import Namespace

from .rnn import RNNQNetwork, TwoHeadedRNNQNetwork

def get_network(args: Namespace, n_outs: int):
    if args.algo == 'rnn':
        return RNNQNetwork(n_actions=n_outs,
                           hidden_size=args.hidden_size,
                           arch=args.arch,
                           value_head_layers=args.value_head_layers)
    elif args.algo == 'multihead_rnn':
        return TwoHeadedRNNQNetwork(n_actions=n_outs,
                                    hidden_size=args.hidden_size,
                                    arch=args.arch,
                                    obs_value_inputs=args.residual_obs_val_input,
                                    value_head_layers=args.value_head_layers)
    else:
        raise NotImplementedError
