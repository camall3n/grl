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
            trial_id = int(info['trial_id'].split('_')[-1]) % 10
            info['trial_id'] = trial_id
            if info['tmax'] < info['tmin']:
                continue
            info['accept_probs'] = info['optimizer_info']['accept_probs']
            info['best_discrep'] = info['optimizer_info']['best_discrep']
            info['temps'] = info['optimizer_info']['temps']
            info['discreps'] = info['optimizer_info']['discreps']
            info['optim_steps'] = np.arange(len(info['discreps']))
            del info['optimizer_info']
            all_results.append(info)
    data = pd.DataFrame(all_results)
    return data

data = load_results('results/discrete/locality01/*/*')
data = data.explode(['accept_probs', 'temps', 'discreps', 'optim_steps'], ignore_index=True)
data.tmax.unique()
subset = data.query('tmax>=1e-2 and tmax<0.3 and tmin <= 1e-6')

sns.relplot(data=subset,
            x='optim_steps',
            y='discreps',
            col='env',
            col_wrap=4,
            facet_kws={'sharey': False},
            hue='tmin',
            style='tmax',
            kind='line',
            units='seed',
            estimator=None)
sns.relplot(data=subset,
            x='optim_steps',
            y='discreps',
            col='env',
            col_wrap=4,
            facet_kws={'sharey': False},
            hue='tmin',
            style='tmax',
            kind='line')

sns.relplot(data=data,
            x='optim_steps',
            y='discreps',
            col='env',
            col_wrap=4,
            facet_kws={'sharey': False},
            hue='tmin',
            style='tmax',
            kind='line')
