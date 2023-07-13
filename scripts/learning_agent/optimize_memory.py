import copy

import numpy as np
from tqdm import tqdm
import optuna

from grl import environment
from grl.mdp import POMDP, MDP
from grl.agent.actorcritic import ActorCritic
from grl.memory.lib import get_memory

#%% Initialize environment

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = POMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.observation_space.n,
                    n_actions=env.action_space.n,
                    gamma=env.gamma,
                    n_mem_entries=0,
                    replay_buffer_size=int(4e6))
agent.set_policy(spec['Pi_phi'][0], logits=False) # policy over non-memory observations

assert agent.policy_logits.shape == (5, 4)
assert agent.memory_logits.shape == (4, 5, 1, 1)

#%% Converge value functions

n_episodes = 20000
for i in tqdm(range(n_episodes)):
    agent.reset_memory_state()
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

#%% Add memory
agent.add_memory()
assert agent.policy_logits.shape == (5 * 2, 4)
assert agent.memory_logits.shape == (4, 5, 2, 2)

#%% Re-converge value functions based on expert memory

agent.set_memory(get_memory(16)) # grab optimal memory for t-maze
discrep = agent.evaluate_memory(n_epochs=1)
preamble_str = f"""Optimal memory function:
{agent.mem_summary()}
Discrep: {discrep}
--------------------------------------------
"""

#%% Compare against noisy versions

# def noisy_mem(mem_fn, alpha=0.1, n_mem_states=2):
#     return (1-alpha) * mem_fn + alpha * np.ones_like(mem_fn) / n_mem_states
#
# agent.set_memory(get_memory(16)) # grab optimal memory for t-maze
# agent.set_memory(noisy_mem(agent.memory_probs, 0.05), logits=False)
# print(agent.mem_summary())
# print(agent.evaluate_memory(n_epochs=1))
#
# print()
# agent.set_memory(get_memory(16)) # grab optimal memory for t-maze
# agent.set_memory(noisy_mem(agent.memory_probs, 0.1), logits=False)
# print(agent.mem_summary())
# print(agent.evaluate_memory(n_epochs=1))
#
# print()
# agent.set_memory(get_memory(16)) # grab optimal memory for t-maze
# agent.set_memory(noisy_mem(agent.memory_probs, 0.2), logits=False)
# print(agent.mem_summary())
# print(agent.evaluate_memory(n_epochs=1))

#%% Optimize memory function

study = agent.optimize_memory(study_name='tmaze_tpe_2k', n_trials=2000, preamble_str=preamble_str)
