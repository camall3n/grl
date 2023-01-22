import copy

import numpy as np
from tqdm import tqdm
import optuna

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory

#%% Initialize environment

spec = environment.load_spec('tmaze_5_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.n_obs,
                    n_actions=env.n_actions,
                    gamma=env.gamma,
                    n_mem_entries=0,
                    replay_buffer_size=int(4e6))

def converge_value_functions(n_episodes=1000):
    for i in tqdm(range(n_episodes)):
        agent.reset()
        obs, _ = env.reset()
        action = agent.act(obs)
        terminal = False
        while not terminal:
            next_obs, reward, terminal, _, info = env.step(action)
            next_action = agent.act(next_obs)

            experience = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'next_obs': next_obs,
                'next_action': next_action,
            }

            agent.update_critic(experience)
            agent.store(experience)

            obs = next_obs
            action = next_action
#%%
n_mem_iterations = 0
for i in range(100):
    print(f'iteration: {i}')
    print(agent.cached_policy_fn)
    converge_value_functions()
    did_change = agent.update_actor()
    if not did_change:
        break

#%%
if agent.n_mem_entries == 0:
    agent.add_memory()
agent.optimize_memory(f'mi1/{n_mem_iterations}', n_jobs=3)
n_mem_iterations += 1
