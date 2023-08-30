from typing import Union, Tuple

from jax import random
import numpy as np
from tqdm import trange

from grl.agent.rnn import RNNAgent
from grl.mdp import MDP, POMDP
from grl.utils.loss import mse
from grl.utils.mdp import all_t_discounted_returns

def eval_episodes(agent: RNNAgent, network_params: dict,
                  env: Union[MDP, POMDP], rand_key: random.PRNGKey,
                  n_episodes: int = 1, test_eps: float = 0.,
                  max_episode_steps: int = 1000)\
        -> Tuple[dict, random.PRNGKey]:
    all_ep_returns = []
    all_ep_discounted_returns = []
    all_ep_q_errs = []

    original_epsilon = agent.eps
    agent.eps = test_eps

    for ep in trange(n_episodes):
        ep_rewards = []
        ep_qs = []

        hs, rand_key = agent.reset(rand_key)

        obs, env_info = env.reset()

        action, rand_key, hs, qs = agent.act(network_params, obs, hs, rand_key)
        action = action.item()
        ep_qs.append(qs[0, 0, action])

        for t in range(max_episode_steps):
            next_obs, reward, done, _, info = env.step(action)

            next_action, rand_key, next_hs, qs = agent.act(network_params, next_obs, hs, rand_key)
            next_action = next_action.item()
            ep_qs.append(qs[0, 0, next_action])

            ep_rewards.append(reward)
            if done:
                break

            # bump time step
            hs = next_hs
            action = next_action

        ep_qs = np.array(ep_qs)
        relevant_ep_qs = ep_qs[:-1]
        episode_rewards = np.array(ep_rewards)

        discounts = np.ones(episode_rewards.shape[0])
        if agent.args.no_gamma_terminal:
            discounts *= env.gamma

        discounts[-1] = 0.

        t_discounted_returns = all_t_discounted_returns(discounts, episode_rewards)
        all_ep_q_errs.append(mse(relevant_ep_qs, t_discounted_returns).item())
        all_ep_returns.append(episode_rewards.sum())
        if agent.args.no_gamma_terminal:
            all_ep_discounted_returns.append(t_discounted_returns[0])

    # reset original epsilon
    agent.eps = original_epsilon
    all_ep_returns = np.array(all_ep_returns, dtype=np.float16)
    all_ep_q_errs = np.array(all_ep_q_errs, dtype=np.float16)

    info = {'episode_returns': all_ep_returns, 'episode_q_errs': all_ep_q_errs}

    if agent.args.no_gamma_terminal:
        all_ep_discounted_returns = np.array(all_ep_discounted_returns, dtype=np.float16)
        info['episode_discounted_returns'] = all_ep_discounted_returns

    return info, rand_key
