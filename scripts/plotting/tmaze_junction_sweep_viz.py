import glob
from pathlib import Path
import os

import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from grl.utils import load_info
from definitions import ROOT_DIR

#%%
def test_mem_matrix(mem_params: jnp.ndarray):
    """
    Tests the memory matrix for t-maze.
    our tolerance is set to 1e-1, which seems high, but should be fine for
    stochastic matrices that we have.
    """
    RIGHT_ACTION = 2
    UP_START_OBS = 0
    DOWN_START_OBS = 1
    CORRIDOR_OBS = 2

    mem_func = softmax(mem_params, axis=-1)
    right_mem_func = mem_func[RIGHT_ACTION]

    # we index by zero here, since memory starts at 0
    right_up_start = right_mem_func[UP_START_OBS, 0]
    right_down_start = right_mem_func[DOWN_START_OBS, 0]

    # we test whether start bits set to different memory states
    def test_start_bits_set(right_up_start: np.ndarray, right_down_start: np.ndarray):
        return np.isclose(np.abs(right_up_start - right_down_start).sum() / 2, 1, atol=1e-1)

    diff_start_bits_set = test_start_bits_set(right_up_start, right_down_start)

    # now we test whether the right corridor memory function is all set or reset
    right_corridor = right_mem_func[CORRIDOR_OBS]

    def test_corridor_hold_or_toggle(right_corridor: np.ndarray):
        is_toggle = np.allclose(right_corridor, np.eye(2)[:, ::-1], atol=1e-1)
        is_hold = np.allclose(right_corridor, np.eye(2), atol=1e-1)
        return is_toggle, is_hold

    is_toggle, is_hold = test_corridor_hold_or_toggle(right_corridor)
    return diff_start_bits_set, is_toggle, is_hold

#%%
def load_sampled_results(pathname: str):
    all_results = []
    results_dirs = glob.glob(pathname)
    for results_dir in results_dirs:
        results_file = results_dir + '/info.npy'
        if not os.path.exists(results_file):
            continue
        info = load_info(results_file)
        final_mem_params = info['final_params']
        diff_start_bits_set, is_toggle, is_hold = test_mem_matrix(final_mem_params)
        is_optimal = diff_start_bits_set and (is_toggle or is_hold)
        result = {
            'policy_up_prob': info['policy_up_prob'],
            'policy_epsilon': info['policy_epsilon'] if 'policy_epsilon' in info else np.nan,
            'trial_id': os.path.basename(results_dir).split('__')[0].split('_')[-1],
            'initial_discrep': info['initial_discrep'],
            'final_discrep': info['final_discrep'],
            'is_optimal': is_optimal
        }
        all_results.append(result)
    data = pd.DataFrame(all_results)
    return data

#%%
def load_analytical_results(pathname: str, use_epsilon=False):
    results_dir = Path(ROOT_DIR, pathname)

    all_results = []
    for results_path in results_dir.iterdir():
        if results_path.suffix != '.npy':
            continue
        info = load_info(results_path)
        grad_info = info['grad_info']
        args = info['args']
        final_mem_params = grad_info['final_params']

        diff_start_bits_set, is_toggle, is_hold = test_mem_matrix(final_mem_params)
        initial_q_discrep, initial_v_discrep = info['initial_discrep']['q'], info[ 'initial_discrep']['v']

        final_mc_vals, final_td_vals = grad_info['final_vals']['mc'], grad_info['final_vals']['td']
        final_v_discrep = (final_mc_vals['v'] - final_td_vals['v'])**2
        final_q_discrep = (final_mc_vals['q'] - final_td_vals['q'])**2

        policy_up_prob = args['tmaze_junction_up_pi']
        policy_epsilon = args['epsilon'] if 'epsilon' in args else np.nan

        p = policy_up_prob
        p_up_policy = np.array([
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [p, (1 - p), 0., 0.],
            [p, (1 - p), 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
        ])

        uniform = np.ones_like(policy_up_prob, dtype=float) / p_up_policy.shape[-1]
        eps_policy = (1 - policy_epsilon) * p_up_policy + policy_epsilon * uniform

        policy = eps_policy if use_epsilon else p_up_policy

        def aggregate_q_discrep(q_discrep, policy):
            return (q_discrep.T * policy).sum(-1).mean()

        result = {
            'policy_up_prob': policy_up_prob,
            'policy_epsilon': policy_epsilon,
            'trial_id': os.path.basename(results_path).split('_s(')[-1].split(')_')[0],
            # 'initial_discrep': initial_v_discrep.mean(),
            # 'final_discrep': final_v_discrep.mean(),
            'initial_discrep': aggregate_q_discrep(initial_q_discrep, policy=policy),
            'final_discrep': aggregate_q_discrep(final_q_discrep, policy=policy),
            'is_optimal': diff_start_bits_set and (is_toggle or is_hold),
        }
        all_results.append(result)
    data = pd.DataFrame(all_results)
    return data

#%%
def plot_sweep(data: pd.DataFrame, x='policy_up_prob', ax=None, title=None, add_colorbar=False):
    y_high = max(data['initial_discrep'].max(), data['final_discrep'].max()) * 1.05
    y_low = 0
    x_high = max(data[x])
    x_low = min(data[x])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    data_seed_avg = data.groupby(x, as_index=False).mean()
    aximg = ax.imshow(
        data_seed_avg['is_optimal'].to_numpy()[None, :],
        vmin=0,
        vmax=1,
        extent=(x_low, x_high, y_low, y_high),
        cmap='RdYlBu',
        aspect='auto',
        alpha=0.5,
    )
    if add_colorbar:
        plt.gcf().colorbar(aximg,
                           label='Optimal memory frequency',
                           ticks=[0.0, 0.5, 1.0],
                           format='%0.1f')
    sns.lineplot(data=data, x=x, y='initial_discrep', color='blue', ax=ax, label='Initial')
    sns.lineplot(data=data, x=x, y='final_discrep', color='black', ax=ax, label='Final')
    ax.legend()
    ax.set_ylabel('Aggregated λ-discrep')
    ax.set_xlabel(x)
    ax.set_title(title)
    plt.gcf().tight_layout()
    plt.gcf().subplots_adjust(right=0.75)

#%%
planning_data = load_analytical_results('results/analytical/tmaze_sweep_junction_pi_old')
learning_data = load_sampled_results(
    'results/sample_based/junction-sweep-sampled-4/tmaze_5_two_thirds_up/*')

np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 14})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_sweep(planning_data, ax=axes[0], title='Planning Agent', add_colorbar=False)
plot_sweep(learning_data, ax=axes[1], title='Learning Agent', add_colorbar=True)

#%%
planning_data = load_analytical_results(pathname='results/analytical/tmaze_pe_sweep_eps', use_epsilon=True)
sns.histplot(data=planning_data, x='policy_epsilon', bins=20)
sns.histplot(data=learning_data, x='policy_epsilon', bins=33)
learning_data = load_sampled_results(
    'results/sample_based/junction-sweep-eps/tmaze_5_two_thirds_up/*')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_sweep(planning_data, ax=axes[0], x='policy_epsilon', title='Planning Agent', add_colorbar=False)
plot_sweep(learning_data, ax=axes[1], x='policy_epsilon', title='Learning Agent', add_colorbar=True)

#%%
fig, ax = plt.subplots()
plt.plot()
sns.lineplot(data=planning_data, ax=ax, x='policy_epsilon', y='initial_discrep', color='blue', linestyle='-')
sns.lineplot(data=learning_data, ax=ax, x='policy_epsilon', y='initial_discrep', color='blue', linestyle='--')
sns.lineplot(data=planning_data, ax=ax, x='policy_epsilon', y='final_discrep', color='black', linestyle='-')
sns.lineplot(data=learning_data, ax=ax, x='policy_epsilon', y='final_discrep', color='black', linestyle='--')
