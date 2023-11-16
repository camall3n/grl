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

def load_results(pathname):
    all_results = []
    results_dirs = glob.glob(pathname)
    for results_dir in results_dirs:
        results_file = results_dir + '/discrete_oracle.json'
        with open(results_file, 'r') as f:
            info = json.load(f)
            info2 = info.copy()
            del info2['optimizer_info']
            trial_id = int(info['trial_id'].split('_')[-1]) % 10
            info['trial_id'] = trial_id
            if info['tmax'] < info['tmin']:
                continue
            info['accept_prob'] = info['optimizer_info']['accept_probs']
            info['best_discrep'] = info['optimizer_info']['best_discrep']
            info['temp'] = info['optimizer_info']['temps']
            info['discrep'] = info['optimizer_info']['discreps']
            info['value_err'] = info['optimizer_info']['value_errs']
            info['optim_step'] = np.arange(len(info['discrep']))
            info['repeat'] = [x for l in [np.ones(info['n_iters']+1) * i for i in info['optimizer_info']['n_repeats']] for x in l]
            info['iter'] = [x for l in [np.ones(info['n_repeats']) * i for i in np.arange(info['n_iters']+1)] for x in l]
            len(info['iter'])
            del info['optimizer_info']
            all_results.append(info)
    data = pd.DataFrame(all_results)
    return data

data = load_results('results/discrete/value-err-01/*/*')
data = data.explode(['accept_prob', 'temp', 'discrep', 'value_err', 'optim_step', 'repeat', 'iter'], ignore_index=True)
data.value_err.unique()

#%%
envs_and_names_and_ylims = [
    ('4x3.95', '4x3', [-0.04, 0.404]),
    ('cheese.95', 'cheese', None),
    ('network', 'network', [-100, 2600]),
    ('paint.95', 'paint', [-0.02, 0.304]),
    ('shuttle.95', 'shuttle', [-1, 6]),
    ('tiger-alt-start', 'tiger', None),
    ('tmaze_5_two_thirds_up', 'tmaze', None),
]
fig, axes = plt.subplots(2, 4, figsize=(12,4))
for (env, name, ylims), ax in zip(envs_and_names_and_ylims, axes.flatten()):
    subset = data.query(f'env=="{env}"').copy()
    subset['value_err'] = pd.to_numeric(subset['value_err'], errors='coerce')
    subset['discrep'] = pd.to_numeric(subset['discrep'], errors='coerce')
    scatter_kwargs = {'s': 10, 'alpha': 0.05, 'marker': 'x'}
    sns.scatterplot(data=subset, x='value_err', y='discrep', ax=ax, **scatter_kwargs)
    sns.regplot(data=subset, x='value_err', y='discrep', scatter=False, ci=False, line_kws={'color': '#dd2244'}, ax=ax)
    ax.set_title(name)
    ax.set_ylim(ylims)
axes.flatten()[-1].axis('off')
for row in axes:
    for i, ax in enumerate(row):
        if i>0:
            ax.set_ylabel('')
fig.tight_layout()
plt.show()
