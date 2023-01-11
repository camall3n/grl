import copy

from jax import nn
import numpy as np
import optuna
from tqdm import tqdm

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory
from grl.utils.math import glorot_init

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.n_obs,
                    n_actions=env.n_actions,
                    gamma=env.gamma,
                    n_mem_entries=0,
                    replay_buffer_size=int(4e6))
agent.set_policy(np.log(spec['Pi_phi'][0] + 1e-6)) # policy over non-memory observations

assert agent.policy_params.shape == (5, 4)
assert agent.memory_params.shape == (4, 5, 1, 1)

#%%
n_episodes = 20000
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
assert agent.policy_params.shape == (5 * 2, 4)
assert agent.memory_params.shape == (4, 5, 2, 2)

#%%

# agent.set_memory(glorot_init(
#     (agent.n_actions, agent.n_obs, agent.n_mem_states, agent.n_mem_states, )
# ))
agent.set_memory(get_memory(16))
# grab optimal memory for t-maze
agent.print_mem_summary()
mem = agent.cached_memory_fn
alpha = 0.2
mem = (1 - alpha) * mem + alpha * np.ones_like(mem) / 2
agent.cached_memory_fn = mem
mem
#%%
agent.evaluate_memory(n_epochs=1)

#%%

study = agent.optimize_memory(n_trials=2000)

#%%
required_params_shape = agent.memory_params.shape[:-1] + (agent.n_mem_values - 1, )

def fill_in_params(required_params):
    params = np.empty_like(agent.memory_params)
    params[:, :, :, :-1] = np.asarray(required_params).reshape(required_params_shape)
    params[:, :, :, -1] = 1 - np.sum(params[:, :, :, :-1], axis=-1)
    return params

required_params = [
    study.best_trial.params[key]
    for key in sorted(study.best_trial.params.keys(), key=lambda x: int(x))
]
params = fill_in_params(required_params)
agent.set_memory(params, logits=False)
agent.print_mem_summary()

#%%
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

plot_contour(study)
plot_edf(study)
plot_optimization_history(study)
plot_parallel_coordinate(study)
plot_param_importances(study)
plot_slice(study)
plt.savefig('foo.png')
