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
len(data.iter)
data = load_results('results/discrete/value-err-logging/*/*')
data = data.explode(['accept_prob', 'temp', 'discrep', 'value_err', 'optim_step', 'repeat', 'iter'], ignore_index=True)
data.value_err.unique()
subset = data.query('value_err<1')

sns.scatterplot(
    data=subset,
    x='value_err',
    y='discrep',
)
# plt.xscale('log')
# plt.yscale('log')
