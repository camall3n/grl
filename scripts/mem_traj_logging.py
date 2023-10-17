from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np


from grl.utils.file_system import load_info
from grl.environment import load_pomdp
from definitions import ROOT_DIR

Step = namedtuple('Step', ['obs', 'mem', 'action', 'reward'])

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
    episodes = 20

    env_name = 'tiger-alt-start'
    env, _ = load_pomdp(env_name, rand_key=rand_state)

    agent_fpath = Path(ROOT_DIR, 'results', 'agent', 'tiger-alt-start_seed(2020)_time(20231016-140310)_a2263f67e30b3282dda096e0fb0144d0.pkl.npy')
    agent = load_info(agent_fpath)

    act = partial(act, n_mem=agent.memory.shape[-1])

    obs_map, action_map = get_mappings(env_name)

    memory = np.array(agent.memory)
    pi = np.array(agent.policy)

    for ep in range(episodes):
        episode_traj = []
        episode_returns = 0

        done = False

        obs, _ = env.reset()
        mem = 0
        while not done:
            action = act(pi, obs, mem)
            next_obs, reward, done, _, info = env.step(action)

            episode_traj.append(Step(obs=obs_map[obs], mem=mem, action=action_map[action], reward=reward))

            mem = memory_step(memory, mem, obs, action)
            obs = next_obs

        print()

    print("here")
