from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import trange

from grl.memory import memory_cross_product
from grl.utils.file_system import load_info
from grl.environment import load_pomdp
from definitions import ROOT_DIR

Step = namedtuple('Step', ['state', 'mem', 'obs', 'action', 'reward'])

def act(policy: np.ndarray, obs: int, mem: int,
        n_mem: int = 4):
    idx = (obs * n_mem) + mem
    return np.random.choice(np.arange(policy.shape[-1]), p=policy[idx])

def memory_step(memory: np.ndarray, obs: int, mem: int, action: int):
    return np.random.choice(np.arange(memory.shape[-1]), p=memory[action, obs, mem])

def get_mappings(env_name: str = 'tiger-alt-start'):
    if env_name == 'tiger-alt-start':
        action_mapping = ['listen', 'open-left', 'open-right']
        obs_mapping = ['init', 'tiger-left', 'tiger-right', 'terminal']
    else:
        raise NotImplementedError
    return obs_mapping, action_mapping

if __name__ == "__main__":
    seed = 2021
    np.random.seed(seed)
    rand_state = np.random.RandomState(seed)
    episodes = 10000

    env_name = 'tiger-alt-start'
    env, _ = load_pomdp(env_name, rand_key=rand_state)

    agent_fpath = Path(ROOT_DIR, 'results', 'agent',
                       'tiger-alt-start_seed(2020)_time(20231018-134915)_4a3ffb1323a55af44906b2082a787957.pkl.npy')
    agent = load_info(agent_fpath)
    mem_env = memory_cross_product(agent.mem_params, env)

    n_mem = agent.memory.shape[-1]
    def convert_m_obs(m_obs: int):
        obs = m_obs // n_mem
        mem = m_obs % n_mem
        return obs, mem

    def convert_to_m_state(state: int, mem: int):
        return (state * n_mem) + mem

    # okay this is a super weird bug. when converting from jax.numpy.array to np.array,
    # we get a roundoff error.

    # we greedify here for testing.
    mem_env.phi = np.array(mem_env.phi)
    mem_env.T = np.array(mem_env.T)
    mem_env.p0 = np.array(mem_env.p0)

    act = partial(act, n_mem=agent.memory.shape[-1])

    obs_map, action_map = get_mappings(env_name)

    memory = np.array(agent.memory)
    pi = np.array(agent.policy)

    discounted_episode_returns = []

    for ep in trange(episodes):
        episode_traj = []
        episode_rewards = []

        done = False
        mem = 0

        obs, _ = env.reset()
        state = env.current_state

        while not done:
            action = act(pi, obs, mem)

            next_obs, reward, done, _, info = env.step(action)

            episode_rewards.append(reward)

            episode_traj.append(Step(obs=obs_map[obs],
                                     mem=mem,
                                     action=action_map[action],
                                     reward=reward,
                                     state=state))


            state = env.current_state

            next_mem = memory_step(memory, obs, mem, action)
            obs = next_obs
            mem = next_mem

        episode_rewards = np.array(episode_rewards)
        episode_discounts = env.gamma ** np.arange(len(episode_rewards))
        discounted_return = np.dot(episode_rewards, episode_discounts)
        discounted_episode_returns.append(discounted_return)


    print(f"average discounted returns: {np.mean(discounted_episode_returns)}")
