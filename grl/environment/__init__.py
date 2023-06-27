from argparse import Namespace

import jax
import numpy as np
from numpy import random

from .rocksample import RockSample
from .spec import load_spec, load_pomdp
from .wrappers import OneHotObservationWrapper, OneHotActionConcatWrapper
import popgym
import gymnasium as gym


def get_popgym_env(args: Namespace, rand_key: random.RandomState = None, **kwargs): 
    # check to see if name exists
    env_names = set([e["id"] for e in popgym.envs.ALL.values()])
    if args.spec not in env_names:
        raise AttributeError(f"spec {args.spec} not found")
    # wrappers fail unless disable_env_checker=True
    env = gym.make(args.spec, disable_env_checker=True)
    env.reset(seed = args.seed)
    env.rand_key = rand_key
    env.gamma = args.gamma

    return env

def get_env(args: Namespace,
            rand_state: np.random.RandomState = None,
            rand_key: jax.random.PRNGKey = None,
            **kwargs):
    # First we check our POMDP specs
    try:
        env, _ = load_pomdp(args.spec, rand_key=rand_state, **kwargs)
    except NotImplementedError:
        # try to load from popgym
        # validate input: we need a custom gamma for popgym args as they don't come with a gamma
        if args.gamma is None:
            raise AttributeError("Can't load non-native environments without passing in gamma!")
        try:
            env, _ = load_pomdp(args.spec, rand_key=rand_state, **kwargs)

        except AttributeError:
            # try to load from popgym
            # validate input: we need a custom gamma for popgym args as they don't come with a gamma
            if args.gamma is None:
                raise AttributeError("Can't load non-native environments without passing in gamma!")
            try:
                env = get_popgym_env(args, rand_key=rand_state, **kwargs)
                
            except AttributeError:
                # don't have anything else implemented
                raise NotImplementedError   

        # TODO: some features are already encoded in a one-hot manner.
        if args.feature_encoding == 'one_hot':
            env = OneHotObservationWrapper(env) 
        
    if args.action_cond == 'cat':
        env = OneHotActionConcatWrapper(env)

    return env
