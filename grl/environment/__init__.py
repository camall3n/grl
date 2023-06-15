from argparse import Namespace

import jax
import numpy as np

from .rocksample import RockSample
from .spec import load_spec, load_pomdp
from .wrappers import OneHotObservationWrapper, OneHotActionConcatWrapper

def get_env(args: Namespace, rand_state: np.random.RandomState = None,
            rand_key: jax.random.PRNGKey = None, **kwargs):
    # First we check our POMDP specs
    try:
        env, _ = load_pomdp(args.spec, rand_key=rand_state, **kwargs)

        # TODO: some features are already encoded in a one-hot manner.
        if args.feature_encoding == 'one_hot':
            env = OneHotObservationWrapper(env)
    except AttributeError:
        if args.spec == 'rocksample':
            env = RockSample(rand_key=rand_key, **kwargs)
        else:
            raise NotImplementedError

    if args.action_cond == 'cat':
        env = OneHotActionConcatWrapper(env)

    return env
