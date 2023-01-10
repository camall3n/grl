import copy

from jax import nn
import numpy as np
import optuna
from tqdm import tqdm

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.n_obs, n_actions=env.n_actions, gamma=env.gamma, n_mem_entries=0)
agent.cached_policy = spec['Pi_phi'][0]# + 1e-6) # policy over non-memory observations

#%%
n_episodes = 10000
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

q_mc_orig = copy.deepcopy(agent.q_mc.q)
q_td_orig = copy.deepcopy(agent.q_td.q)

#%%
