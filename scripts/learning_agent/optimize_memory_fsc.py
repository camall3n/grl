from tqdm import tqdm
from matplotlib import pyplot as plt

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.finite_state_controller import FiniteStateController

#%% Initialize environment

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = FiniteStateController(n_obs=env.n_obs,
                              n_actions=env.n_actions,
                              n_nodes=4)
#%% Run policy

all_returns = []
n_episodes = 20000
for i in tqdm(range(n_episodes)):
    episode_return = 0

    agent.reset()
    obs, _ = env.reset()
    action = agent.act(obs)
    terminal = False
    while not terminal:
        curr_node = agent.curr_node
        next_obs, reward, terminal, _, info = env.step(action)
        next_action = agent.act(next_obs)

        experience = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'terminal': terminal,
            'next_obs': next_obs,
            'next_action': next_action,
            'curr_node': curr_node,
            'next_node': agent.curr_node, # agent.curr_node updates during agent.act() above
        }

        # print(experience)
        # print()
        agent.store(experience)
        agent.update_weights(experience)

        episode_return += reward
        obs = next_obs
        action = next_action


    all_returns.append(episode_return)
    print()
    print('------')

plt.plot(all_returns)
plt.show()
#%% Optimize memory
# agent.optimize_memory()
