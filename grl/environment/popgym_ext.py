"""
Helper file for accessing popgym environments.
"""
from argparse import Namespace
import gymnasium as gym
from numpy import random 
from .wrappers import GammaTerminalWrapper

import popgym

def get_popgym_env(args: Namespace, rand_key: random.RandomState = None, **kwargs): 
    # check to see if name exists
    env_names = set([e["id"] for e in popgym.envs.ALL.values()])
    if args.spec not in env_names:
        raise AttributeError(f"spec {args.spec} not found")
    # wrappers fail unless disable_env_checker=True
    env = gym.make(args.spec, disable_env_checker=True)
    env.reset(seed = args.seed)
    env.rand_key = rand_key

    # make it compatible with gamma_terminal API
    # this is necessary to use with our Trainer
    wrapped_env = GammaTerminalWrapper(env, args.gamma)

    return wrapped_env
