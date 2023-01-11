import copy

from jax import nn
import numpy as np
import optuna
from tqdm import tqdm

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.n_obs, n_actions=env.n_actions, gamma=env.gamma, n_mem_entries=0, replay_buffer_size=int(4e6))
agent.set_policy(np.log(spec['Pi_phi'][0] + 1e-6)) # policy over non-memory observations

#%%
n_episodes = 30000
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
d0 = np.abs(q_mc_orig - q_td_orig).sum()

#%%

agent.add_memory(n_mem_entries=1)
agent.set_memory(get_memory(16)) # grab optimal memory for t-maze

# #%%
# tqdm()
# agent.reset()
# agent.q_mc.reset()
# agent.q_td.reset()
# for experience in agent.replay.memory:
#     e = experience.copy()
#     del e['_index_']
#     agent.memory
#     agent.prev_memory
#     agent.cached_memory_fn[e['action'], e['obs']].round(1)
#     agent.step_memory(e['obs'], e['action'])
#     agent.augment_obs(0, 0)
#     agent.augment_experience(e)
#     agent.update_critic(e)
#     if experience['terminal']:
#         agent.reset()
# return np.abs(agent.q_mc.q - agent.q_td.q).mean()



#%%
agent.evaluate_memory(n_epochs=3)
d0
d1 = np.abs(agent.q_mc.q - agent.q_td.q).sum()


q_mc_orig
q_td_orig

agent.q_mc.q.round(3)

agent.q_td.q.round(3)
