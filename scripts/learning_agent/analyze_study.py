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
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory

env_name = 'tmaze_5_two_thirds_up'
results_dirs = glob.glob('results/sample_based/exp11*')

def load_study(experiment_name, env_name, seed):
    study_name = f'{experiment_name}/{env_name}/{seed}'
    study_dir = f'./results/sample_based/{study_name}'
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(f"{study_dir}/study.journal"))
    study = optuna.load_study(
        study_name=f'{study_name}',
        storage=storage,
    )
    return study

def study_to_pandas(study, experiment_name, env_name, seed):
    df = pd.DataFrame([{
        'trial_id': t.number,
        'experiment_name': experiment_name,
        'env_name': env_name,
        'seed': seed,
        'state': str(t.state).replace('TrialState.', ''),
        'value': t.values[0] if t.values is not None and t.values != [] else np.nan,
        'datetime_start': t.datetime_start,
        'datetime_complete': t.datetime_complete,
        'duration': (t.datetime_complete - t.datetime_start) if (t.datetime_complete is not None) else None,
    } for t in study.trials])
    complete_trials = df.query('state=="COMPLETE"')
    values_by_trial_id = complete_trials.sort_values(by='trial_id')['value'].tolist()
    best_value_by_trial_id = np.minimum.accumulate(values_by_trial_id)
    complete_trials = complete_trials.assign(best_value=best_value_by_trial_id)
    return complete_trials

dfs = []
for results_dir in tqdm(results_dirs):
    experiment_name = os.path.basename(results_dir)
    envs = sorted(list(map(os.path.basename, glob.glob(f'{results_dir}/*'))))
    assert env_name in envs
    seeds = sorted(list(map(os.path.basename, glob.glob(f'{results_dir}/{env_name}/*'))))
    for seed in seeds:
        study = load_study(experiment_name, env_name, seed)
        data = study_to_pandas(study, experiment_name, env_name, seed)
        dfs.append(data)

data = pd.concat(dfs, ignore_index=True)

#%%
# data.to_pickle('results/sample_based/exp11_summary.pkl')

#%%
sns.lineplot(data=data, x='trial_id', y='best_value', hue='experiment_name', units='seed', estimator=None)

#%%
data.query("experiment_name=='exp11-cmaes-startup-tpe'").sort_values(by='best_value')

#%%
study = load_study('exp11-cmaes-startup-tpe', env_name, '10')
params = [
    study.best_trial.params[key] for key in sorted(study.best_trial.params.keys(), key=int)
]
spec = environment.load_spec(env_name, memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.n_obs,
                    n_actions=env.n_actions,
                    gamma=env.gamma,
                    n_mem_entries=0,
                    replay_buffer_size=int(4e6))
agent.reset_policy()
# agent.set_policy(spec['Pi_phi'][0], logits=False) # policy over non-memory observations
agent.add_memory()
agent.set_memory(agent.fill_in_params(params), logits=False)
# print(agent.cached_memory_fn.round(3))
mem = agent.cached_memory_fn.round(3)
print(str(np.concatenate((mem[2, 0], mem[2, 1], mem[2, 2]), axis=-1)))
print()

#%%
plot_optimization_history(study)
plt.title(f'Optimization History ({experiment_name})')
plt.xlim([-20, 1000])
plt.ylim([-0.01, 1.0])
plt.legend(loc='upper left', facecolor='white', framealpha=0.9)
plt.savefig(f'opt-history.png')
plt.close()

study.add_trial()
