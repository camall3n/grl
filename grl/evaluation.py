from typing import Union, Tuple, List

from jax import random
import numpy as np
from tqdm import trange

from grl.agent.rnn import RNNAgent
from grl.mdp import MDP, POMDP
from grl.utils.data import compress_episode_rewards

def eval_episodes(agent: RNNAgent, network_params: dict,
                  env: Union[MDP, POMDP], rand_key: random.PRNGKey,
                  n_episodes: int = 1, test_eps: float = 0.,
                  max_episode_steps: int = 1000)\
        -> Tuple[dict, random.PRNGKey]:
    all_ep_rews = []
    all_ep_qs = []

    original_epsilon = agent.eps
    agent.eps = test_eps

    for ep in trange(n_episodes):
        ep_rews = []
        ep_qs = []

        hs, rand_key = agent.reset(rand_key)

        obs, env_info = env.reset()

        action, rand_key, hs, qs = agent.act(network_params, obs, hs, rand_key)
        action = action.item()
        ep_qs.append(qs[0, 0, action])

        for t in range(max_episode_steps):
            # TODO: we have gamma_terminal as False here. Is that what we want?
            next_obs, reward, done, _, info = env.step(action, gamma_terminal=False)

            next_action, rand_key, next_hs, qs = agent.act(network_params, next_obs, hs, rand_key)
            next_action = next_action.item()
            ep_qs.append(qs[0, 0, next_action])

            ep_rews.append(reward)
            if done:
                break

            # bump time step
            hs = next_hs
            action = next_action

        all_ep_rews.append(compress_episode_rewards(ep_rews))
        all_ep_qs.append(np.array(ep_qs, dtype=np.half))

    # reset original epsilon
    agent.eps = original_epsilon

    info = {'episode_rewards': all_ep_rews, 'episode_qs': all_ep_qs}

    return info, rand_key
