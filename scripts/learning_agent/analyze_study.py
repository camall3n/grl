import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from grl import environment
from grl.mdp import POMDP, MDP
from grl.agent.actorcritic import ActorCritic
from grl.memory.lib import get_memory
from scripts.learning_agent.optuna_to_pandas import load_study

#%%

data = pd.read_pickle('results/sample_based/exp12-tmaze_2_two_thirds_up-summary.pkl')
subset = data
# subset=subset.query("experiment_name=='exp11-cmaes-sigma0'")
# subset['sigma'] = list(map(lambda x: x.split('_')[-1], subset['seed']))
# subset['seed'] = list(map(lambda x: int(x.split('_')[-4]) % 5, subset['seed']))
sns.lineplot(data=subset, x='trial_id', y='best_value') #, units='seed', estimator=None)
plt.savefig('foo.png')

#%%
env_name = 'tmaze_2_two_thirds_up'
study = load_study('foooo', env_name, '1')
plot_optimization_history(study)
plt.title(f'Optimization History ({experiment_name})')
plt.xlim([-20, 1000])
plt.ylim([-0.01, 1.0])
plt.legend(loc='upper left', facecolor='white', framealpha=0.9)
plt.savefig(f'opt-history.png')
plt.close()

#%%

params = [study.best_trial.params[key] for key in sorted(study.best_trial.params.keys(), key=int)]
spec = environment.load_spec(env_name, memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = POMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.observation_space.n,
                    n_actions=env.action_space.n,
                    gamma=env.gamma,
                    n_mem_entries=0,
                    replay_buffer_size=int(4e6))
agent.reset_policy()
# agent.set_policy(spec['Pi_phi'][0], logits=False) # policy over non-memory observations
agent.add_memory()
agent.set_memory(agent.fill_in_params(params), logits=False)
# print(agent.memory_probs.round(3))
mem = agent.memory_probs.round(3)
print(str(np.concatenate((mem[2, 0], mem[2, 1], mem[2, 2]), axis=-1)))
print()
