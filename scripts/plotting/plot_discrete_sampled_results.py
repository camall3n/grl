import glob
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

results_dir = 'results/sample_based/exp31-sampled-2m/tmaze_5_two_thirds_up/1'

def load_sampled_results(pathname: str, use_epsilon: bool = False):
    all_results = []
    results_dirs = glob.glob(pathname)
    for results_dir in results_dirs:
        results_file = results_dir + '/info.npy'
        if not os.path.exists(results_file):
            continue
        info = load_info(results_file)

        env_name = results_dir.split('/')[-2]
        policy_probs = info['initial_policy_history'][0][-1]
        initial_value = get_start_obs_value(env_name, policy_probs)
        scale_low = calibrations_dict[env_name]['init_improvement_perf']
        scale_high = calibrations_dict[env_name]['compare_to_perf']
        scaled_initial_value = (initial_value - scale_low) / (scale_high - scale_low)

        final_memory_logits = info['final_params']
        policy_probs = info['mem_opt_policy_history'][-1][-1]
        final_value = get_start_obs_value(env_name, policy_probs, final_memory_logits)
        scaled_final_value = (final_value - scale_low) / (scale_high - scale_low)

        # is_optimal, mem_info = test_mem_matrix(final_mem_params, test_preserving=use_epsilon)
        try:
            initial_discrep = info['initial_discrep']
        except KeyError:
            initial_discrep = load_info(
                results_dir + '/initial_mem_info.pkl.npy')['sample_based']['discrepancy_loss']
        result = {
            'env': maybe_spec_map(env_name),
            # 'policy_up_prob': info['policy_up_prob'],
            'policy_epsilon': info['policy_epsilon'] if 'policy_epsilon' in info else np.nan,
            'trial_id': results_dir.split('/')[-1],
            'initial_discrep': float(initial_discrep),
            'final_discrep': float(info['final_discrep']),
            'initial_value': float(initial_value),
            'final_value': float(final_value),
            'scaled_initial_value': float(scaled_initial_value),
            'scaled_final_value': float(scaled_final_value),
            # 'is_optimal': is_optimal
        }
        all_results.append(result)
    data = pd.DataFrame(all_results)
    return data

data = load_sampled_results('results/sample_based/exp31-sampled-2m/*/*').sort_values(by='env')

#%%
sns.barplot(data=calibrations_data, x='env', y='scaled_final_value')
xlim = plt.gca().get_xlim()
plt.hlines(0, *xlim, 'k', '--')
plt.hlines(1, *xlim, 'k', '--')
plt.ylim([-0.25, 1.25])
plt.title('Planning agent')
#%%
sns.barplot(data=data, x='env', y='scaled_final_value')
xlim = plt.gca().get_xlim()
plt.hlines(0, *xlim, 'k', '--')
plt.hlines(1, *xlim, 'k', '--')
plt.ylim([-0.25, 1.25])
plt.title('Learning agent (2M steps)')
