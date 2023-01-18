from tqdm import tqdm

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.finite_state_controller import FiniteStateController

#%% Initialize environment

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = FiniteStateController(n_obs=env.n_obs,
                              n_actions=env.n_actions,
                              n_nodes=2)
#%% Run policy

n_episodes = 1000
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

        agent.store(experience)

        obs = next_obs
        action = next_action
