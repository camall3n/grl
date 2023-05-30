from typing import Union, Tuple, List

from jax import random
import numpy as np
from tqdm import trange

from grl.agent.rnn import RNNAgent
from grl.mdp import MDP, AbstractMDP
from grl.utils.data import one_hot

def test_episodes(agent: RNNAgent, network_params: dict,
                  env: Union[MDP, AbstractMDP], rand_key: random.PRNGKey,
                  n_episodes: int = 1, test_eps: float = 0.,
                  action_cond: str = 'cat', max_episode_steps: int = 1000)\
        -> Tuple[dict, random.PRNGKey]:
    all_ep_rews = []
    all_ep_qs = []

    original_epsilon = agent.eps
    agent.eps = test_eps

    one_hot_obses = isinstance(env, MDP) or isinstance(env, AbstractMDP)

    for ep in trange(n_episodes):
        ep_rews = []
        ep_qs = []

        hs, rand_key = agent.reset(rand_key)

        obs, env_info = env.reset()
        if one_hot_obses:
            obs = one_hot(obs, env.n_obs)

        # Action conditioning for t=-1 action
        if action_cond == 'cat':
            action_encoding = np.zeros(env.n_actions)
            obs = np.concatenate([obs, action_encoding], axis=-1)

        action, rand_key, hs, qs = agent.act(network_params, obs, hs, rand_key)
        ep_qs.append(qs)
        action = action.item()

        for t in range(max_episode_steps):
            # TODO: we have gamma_terminal as False here. Is that what we want?
            next_obs, reward, done, _, info = env.step(action, gamma_terminal=False)
            if one_hot_obses:
                next_obs = one_hot(next_obs, env.n_obs)

            # Action conditioning
            if action_cond == 'cat':
                action_encoding = one_hot(action, env.n_actions)
                next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

            next_action, rand_key, next_hs, qs = agent.act(network_params, next_obs, hs, rand_key)
            ep_qs.append(qs)
            next_action = next_action.item()

            ep_rews.append(reward)
            if done:
                break

            # bump time step
            hs = next_hs
            action = next_action
        all_ep_rews.append(ep_rews)
        all_ep_qs.append(ep_qs)


    # reset original epsilon
    agent.eps = original_epsilon

    info = {
        'episode_rewards': all_ep_rews,
        'episode_qs': all_ep_qs
    }

    return info, rand_key
