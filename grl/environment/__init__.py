from argparse import Namespace

from numpy import random

from .spec import load_spec, load_pomdp
from .wrappers import OneHotObservationWrapper, OneHotActionConcatWrapper

def get_env(args: Namespace, rand_key: random.RandomState = None, **kwargs):
    # First we check our POMDP specs
    try:
        env, _ = load_pomdp(args.spec, rand_key=rand_key, **kwargs)
    except AttributeError:
        # we haven't implemented anything other than the POMDP specs yet!
        raise NotImplementedError

    if args.feature_encoding == 'one_hot':
        env = OneHotObservationWrapper(env)

    if args.action_cond == 'cat':
        env = OneHotActionConcatWrapper(env)

    return env
