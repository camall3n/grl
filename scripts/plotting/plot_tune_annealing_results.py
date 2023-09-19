import glob
import json
from pathlib import Path
import os

from jax.config import config
import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from grl.environment import load_spec
from grl.mdp import MDP, POMDP
from grl.utils import load_info
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.discrete_search import generate_hold_mem_fn
from grl.memory import memory_cross_product
from definitions import ROOT_DIR

config.update('jax_platform_name', 'cpu')

def maybe_spec_map(id: str):
    spec_map = {
        '4x3.95': '4x3',
        'cheese.95': 'cheese',
        'paint.95': 'paint',
        'shuttle.95': 'shuttle',
        'example_7': 'ex.7',
        'tmaze_5_two_thirds_up': 'tmaze',
        'tiger-alt-start': 'tiger'
    }
    if id not in spec_map:
        return id
    return spec_map[id]

calibrations_data = pd.read_csv('results/discrete/all_pomdps_means.csv', index_col='spec')
calibrations_dict = calibrations_data.to_dict('index')
calibrations_data = calibrations_data.reset_index()
scale_low = calibrations_data['init_improvement_perf']
scale_high = calibrations_data['compare_to_perf']
scaled_final_values = (calibrations_data['final_mem_perf'] - scale_low) / (scale_high - scale_low)
calibrations_data['scaled_final_value'] = scaled_final_values
calibrations_data['env'] = calibrations_data['spec'].map(maybe_spec_map)

def get_start_obs_value(env_name, policy_probs, memory_logits=None):
    spec = load_spec(env_name, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = POMDP(mdp, spec['phi'])

    if memory_logits is not None:
        env = memory_cross_product(memory_logits, env)
        # policy_probs = policy_probs.repeat(memory_logits.shape[-1], axis=0)

    mc_value_fn, _ = lstdq_lambda(policy_probs, env, lambda_=0.99999)
    return (mc_value_fn @ (env.p0 @ env.phi)).item()

def load_results(pathname):
    all_results = []
    results_dirs = glob.glob(pathname)
    for results_dir in results_dirs:
        results_file = results_dir + '/discrete_oracle.json'
        with open(results_file, 'r') as f:
            info = json.load(f)
            trial_id = int(info['trial_id'].split('_')[-1]) % 10
            info['trial_id'] = trial_id
            scales = calibrations_dict[info['env']]
            info['scaled_final_value'] = (info['end_value'] - scales['init_improvement_perf']) / (
                scales['compare_to_perf'] - scales['init_improvement_perf'])
            if info['tmax'] < info['tmin']:
                continue
            del info['optimizer_info']
            all_results.append(info)
    data = pd.DataFrame(all_results)
    return data

#%%
# data = load_results('results/discrete/tune07-1repeats*/*/*')
data = load_results('results/discrete/locality01/*/*')

n_tmin = len(data.tmin.unique())
n_tmax = len(data.tmax.unique())
n_prog = len(data.progress_fraction_at_tmin.unique())
n_seed = len(data.seed.unique())
n_envs = len(data.env.unique())

param_counts = [n_tmin, n_tmax, n_prog, n_seed, n_envs]
n_runs = np.prod(param_counts)
str_counts = ' x '.join(list(map(str, param_counts)))
print(f'Loaded {n_runs} runs: {str_counts}')

#%%
sns.barplot(data=data, x='env', y='scaled_final_value')
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()

#%%
progress_fraction_at_tmin = 0.7

def histplot(env, x, y, z, ax):
    vmin = calibrations_dict[env]['init_improvement_perf']
    vmax = calibrations_dict[env]['final_mem_perf']
    subset = data.query(f'env=="{env}" and progress_fraction_at_tmin=={progress_fraction_at_tmin}')
    best_discrep = subset.groupby([y, x])['best_discrep'].idxmin()
    data_pivot = subset.loc[best_discrep].pivot(index=y, columns=x, values=z)

    # Create the heatmap
    # plt.figure(figsize=(10, 8))
    sns.heatmap(data_pivot, ax=ax, vmin=vmin, vmax=vmax)

data.columns
fig, axes = plt.subplots(2, 4, figsize=(16, 12))
for env, ax in zip(data.env.unique(), axes.flatten()):
    histplot(env, x='tmax', y='tmin', z='end_value', ax=ax)
    ax.set_title(f'{env}')

    plt.suptitle(f'progress_fraction_at_tmin {progress_fraction_at_tmin}')
plt.tight_layout()
plt.show()

#%%
def histplot(frac, x, y, z, ax):
    env = 'shuttle.95'
    vmin = calibrations_dict[env]['init_improvement_perf']
    vmax = calibrations_dict[env]['final_mem_perf']
    subset = data.query(f'env=="{env}" and progress_fraction_at_tmin=={frac}')
    best_discrep = subset.groupby([y, x])['best_discrep'].idxmin()
    data_pivot = subset.loc[best_discrep].pivot(index=y, columns=x, values=z)

    # Create the heatmap
    # plt.figure(figsize=(10, 8))
    sns.heatmap(data_pivot, ax=ax, vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
for frac, ax in zip(sorted(data.progress_fraction_at_tmin.unique()), axes.flatten()):
    histplot(frac, x='tmax', y='tmin', z='end_value', ax=ax)
    ax.set_title(f'shuttle ({frac})')

plt.tight_layout()
plt.show()

#%%
def histplot(frac, x, y, z, ax):
    env = 'network'
    vmin = calibrations_dict[env]['init_improvement_perf']
    vmax = calibrations_dict[env]['final_mem_perf']
    subset = data.query(
        f'study_name=="tune07-1repeats" and env=="{env}" and progress_fraction_at_tmin=={frac}')
    best_discrep = subset.groupby([y, x])['best_discrep'].idxmin()
    data_pivot = subset.loc[best_discrep].pivot(index=y, columns=x, values=z)

    # Create the heatmap
    # plt.figure(figsize=(10, 8))
    sns.heatmap(data_pivot, ax=ax, vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
for frac, ax in zip(sorted(data.progress_fraction_at_tmin.unique()), axes.flatten()):
    histplot(frac, x='tmax', y='tmin', z='end_value', ax=ax)
    ax.set_title(f'network ({frac})')

plt.tight_layout()
plt.show()

#%%
df = data #.query('study_name=="tune07-1repeats-ctd"')

def histplot(frac, x, y, z, ax):
    env = 'network'
    vmin = calibrations_dict[env]['init_improvement_perf']
    vmax = calibrations_dict[env]['final_mem_perf']
    subset = df.query(f'env=="{env}" and progress_fraction_at_tmin=={frac}')
    # best_discrep = subset.groupby([y, x])['best_discrep'].idxmin()
    data_pivot = subset.groupby([y, x])[z].mean().reset_index().pivot(index=y, columns=x, values=z)

    # Create the heatmap
    # plt.figure(figsize=(10, 8))
    sns.heatmap(data_pivot, ax=ax, vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
for frac, ax in zip(sorted(df.progress_fraction_at_tmin.unique()), axes.flatten()):
    histplot(frac, x='tmax', y='tmin', z='end_value', ax=ax)
    ax.set_title(f'network ({frac})')

plt.tight_layout()
plt.show()

#%%
df = data.query('study_name=="tune07-1repeats"')

def histplot(frac, x, y, z, ax):
    env = 'network'
    vmin = calibrations_dict[env]['init_improvement_perf']
    vmax = calibrations_dict[env]['final_mem_perf']
    subset = df.query(f'env=="{env}" and progress_fraction_at_tmin=={frac}')
    # best_discrep = subset.groupby([y, x])['best_discrep'].idxmin()
    data_pivot = subset.groupby([y, x])[z].mean().reset_index().pivot(index=y, columns=x, values=z)

    # Create the heatmap
    # plt.figure(figsize=(10, 8))
    sns.heatmap(data_pivot, ax=ax, vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
for frac, ax in zip(sorted(df.progress_fraction_at_tmin.unique()), axes.flatten()):
    histplot(frac, x='tmax', y='tmin', z='end_value', ax=ax)
    ax.set_title(f'network ({frac})')

plt.tight_layout()
plt.show()
