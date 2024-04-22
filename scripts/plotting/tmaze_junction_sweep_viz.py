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

from grl.utils import load_info
from definitions import ROOT_DIR

config.update('jax_platform_name', 'cpu')

#%%
# count-based weighting results
# planning_up_prob_filename = 'results/analytical/tmaze_sweep_junction_pi_2023-02-21'
# learning_up_prob_filename = 'results/sample_based/junction-up-prob-lambda999-6/tmaze_5_two_thirds_up/*'
# planning_eps_filename = 'results/analytical/tmaze_sweep_eps_2023-02-21'
# learning_eps_filename = 'results/sample_based/junction-sweep-eps-lambda999-03/tmaze_5_two_thirds_up/*'

#%%
# importance sampling results
planning_up_prob_filename = str(Path(ROOT_DIR, 'results/tmaze_sweep_junction_pi_obs_space'))
learning_up_prob_filename = str(
    Path(ROOT_DIR, 'results/sample_based/sweep-up-prob-imp-samp-7/tmaze_5_two_thirds_up/*'))
planning_eps_filename = str(Path(ROOT_DIR, 'results/tmaze_sweep_eps_obs_space'))
# learning_eps_filename = 'results/sample_based/sweep-eps-imp-samp-04/tmaze_5_two_thirds_up/*'
learning_eps_oracle_filename = str(
    Path(ROOT_DIR, 'results/sample_based/sweep-eps-05-analytical-optuna/tmaze_5_two_thirds_up/*'))
learning_eps_filename = str(
    Path(ROOT_DIR, 'results/sample_based/sweep-eps-06/tmaze_5_two_thirds_up/*'))

#%%
def test_mem_matrix(mem_params: jnp.ndarray, test_preserving: bool = True, mse: bool = False):
    """
    Tests the memory matrix for t-maze.
    our tolerance is set to 1e-1, which seems high, but should be fine for
    stochastic matrices that we have.
    """
    NORTH, SOUTH, EAST, WEST = list(range(4))
    UP_START, DOWN_START, CORRIDOR, JUNCTION = list(range(4))

    mem_func = softmax(mem_params, axis=-1)
    right_mem_func = mem_func[EAST]

    # we index by zero here, since memory starts at 0
    right_up_start = right_mem_func[UP_START, 0]
    right_down_start = right_mem_func[DOWN_START, 0]

    # we test whether start bits set to different memory states
    def test_start_bits_set(right_up_start: np.ndarray, right_down_start: np.ndarray):
        res = np.abs(right_up_start - right_down_start).sum() / 2
        if mse:
            return (res - 1)**2
        else:
            return np.isclose(res, 1, atol=1e-1)

    diff_start_bits_set = test_start_bits_set(right_up_start, right_down_start)

    # now we test whether the right corridor memory function is all set or reset
    right_corridor = right_mem_func[CORRIDOR]

    def test_corridor_hold_or_toggle(right_corridor: np.ndarray):
        if mse:
            is_toggle = ((right_corridor - np.eye(2)[:, ::-1])**2).mean().item()
            is_hold = ((right_corridor - np.eye(2))**2).mean().item()
            return is_toggle, is_hold
        else:
            is_toggle = np.allclose(right_corridor, np.eye(2)[:, ::-1], atol=1e-1)
            is_hold = np.allclose(right_corridor, np.eye(2), atol=1e-1)
            return is_toggle, is_hold

    def test_memory_preserving(mem: np.ndarray):
        # everything that's not part of the optimal policy
        # ie. check for memory preserving functions
        corridor_bumps = mem[[NORTH, SOUTH], CORRIDOR]
        corridor_left = mem[[WEST], CORRIDOR]

        junction_bumps = mem[[EAST], JUNCTION]
        junction_left = mem[[WEST], JUNCTION]
        all_checks = {
            'corridor_bumps': corridor_bumps,
            'corridor_left': corridor_left,
            'junction_bumps': junction_bumps,
            # 'junction_left': junction_left
        }
        if mse:
            mses = []
            for check in all_checks.values():
                mses.append(((check - np.eye(2))**2).mean().item())
            return sum(mses) / len(mses), all_checks
        else:
            is_preserving = np.allclose(np.concatenate(tuple(all_checks.values()), axis=0),
                                        np.eye(2),
                                        atol=1e-1)
            return is_preserving, all_checks

    is_toggle, is_hold = test_corridor_hold_or_toggle(right_corridor)

    if mse:
        is_optimal = 0.5 * diff_start_bits_set + 0.5 * min(is_toggle, is_hold)
    else:
        is_optimal = diff_start_bits_set and (is_toggle or is_hold)

    additional_info = {
        'diff_start_bits_set': diff_start_bits_set,
        'is_toggle': is_toggle,
        'is_hold': is_hold
    }
    if test_preserving:
        is_preserving, all_preserving_checks = test_memory_preserving(mem_func)

        if mse:
            is_optimal = 1 / 3 * (diff_start_bits_set + min(is_toggle, is_hold) +
                                  2 * is_preserving)
        else:
            is_optimal = is_optimal and is_preserving
        additional_info.update(all_preserving_checks)
        additional_info['is_preserving'] = is_preserving

    return is_optimal, additional_info

#%%
def load_sampled_results(pathname: str, use_epsilon: bool = False):
    all_results = []
    results_dirs = glob.glob(pathname)
    for results_dir in results_dirs:
        results_file = results_dir + '/info.npy'
        if not os.path.exists(results_file):
            continue
        info = load_info(results_file)

        final_mem_params = info['final_params']
        is_optimal, mem_info = test_mem_matrix(final_mem_params, test_preserving=use_epsilon)
        try:
            initial_discrep = info['initial_discrep']
        except KeyError:
            initial_discrep = load_info(
                results_dir + '/initial_mem_info.pkl.npy')['sample_based']['discrepancy_loss']
        result = {
            'policy_up_prob': info['policy_up_prob'],
            'policy_epsilon': info['policy_epsilon'] if 'policy_epsilon' in info else np.nan,
            'trial_id': os.path.basename(results_dir).split('__')[0].split('_')[-1],
            'initial_discrep': float(initial_discrep),
            'final_discrep': float(info['final_discrep']),
            'is_optimal': is_optimal
        }
        all_results.append(result)
    data = pd.DataFrame(all_results)
    return data

#%%
def load_analytical_results(pathname: str, use_epsilon=False):
    results_dir = Path(ROOT_DIR, pathname)

    def aggregate_q_discrep(q_discrep, policy):
        return (q_discrep.T * policy).sum(-1).mean()

    all_results = []
    for results_path in results_dir.iterdir():
        if results_path.suffix != '.npy':
            continue
        info = load_info(results_path)
        args = info['args']

        policy_up_prob = args['tmaze_junction_up_pi']
        policy_epsilon = args['epsilon'] if 'epsilon' in args else np.nan
        if 'grad_info' in info:
            grad_info = info['grad_info']
            final_mem_params = grad_info['final_params']

            is_optimal, additional_checks = test_mem_matrix(final_mem_params,
                                                            test_preserving=use_epsilon)
            initial_q_discrep, initial_v_discrep = info['initial_discrep']['q'], info[
                'initial_discrep']['v']
            final_mc_vals, final_td_vals = grad_info['final_vals']['mc'], grad_info['final_vals'][
                'td']
            final_v_discrep = (final_mc_vals['v'] - final_td_vals['v'])**2
            final_q_discrep = (final_mc_vals['q'] - final_td_vals['q'])**2

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

            initial_aggregated_q_discrep = aggregate_q_discrep(initial_q_discrep, policy=policy)
            final_aggregated_q_discrep = aggregate_q_discrep(final_q_discrep, policy=policy)
        else:
            grad_info = info['logs']
            agent_info = load_info(results_path.parent / 'agent' / f'{results_path.stem}.pkl.npy')
            final_mem_params = agent_info.mem_params
            is_optimal, additional_checks = test_mem_matrix(final_mem_params,
                                                            test_preserving=use_epsilon)
            initial_q_discrep = grad_info['initial_mem_stats']['discrep'].item()
            final_q_discrep = grad_info['final_mem_stats']['discrep'].item()
            initial_aggregated_q_discrep = initial_q_discrep
            final_aggregated_q_discrep = final_q_discrep

        result = {
            'policy_up_prob': policy_up_prob,
            'policy_epsilon': policy_epsilon,
            'trial_id': os.path.basename(results_path).split('_s(')[-1].split(')_')[0],
            'seed': args['seed'],
            # 'initial_discrep': initial_v_discrep.mean(),
            # 'final_discrep': final_v_discrep.mean(),
            'initial_discrep': initial_aggregated_q_discrep,
            'final_discrep': final_aggregated_q_discrep,
            'is_optimal': is_optimal,
            'additional_checks': additional_checks
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
planning_data = load_analytical_results(pathname=planning_up_prob_filename)
learning_data = load_sampled_results(learning_up_prob_filename)

np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 14})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_sweep(planning_data, ax=axes[0], title='Planning Agent', add_colorbar=False)
axes[0].set_xlim(0, 0.5)
plot_sweep(learning_data, ax=axes[1], title='Learning Agent', add_colorbar=True)

#%%
planning_data = load_analytical_results(pathname=planning_eps_filename, use_epsilon=True)
learning_data_oracle = load_sampled_results(pathname=learning_eps_oracle_filename)
learning_data = load_sampled_results(learning_eps_filename)

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)
plot_sweep(planning_data,
           ax=axes[0],
           x='policy_epsilon',
           title='SGD (analytical)',
           add_colorbar=False)
plot_sweep(learning_data_oracle,
           ax=axes[1],
           x='policy_epsilon',
           title='Optuna (oracle)',
           add_colorbar=False)
plot_sweep(learning_data,
           ax=axes[2],
           x='policy_epsilon',
           title='Optuna (sampled)',
           add_colorbar=True)

#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_sweep(planning_data,
           ax=axes[0],
           x='policy_epsilon',
           title='Planning Agent',
           add_colorbar=False)
plot_sweep(learning_data,
           ax=axes[1],
           x='policy_epsilon',
           title='Learning Agent',
           add_colorbar=True)
#%%
planning_data[planning_data['policy_epsilon']] < 0.2

#%%
fig, ax = plt.subplots()
plt.plot()
sns.lineplot(data=planning_data,
             ax=ax,
             x='policy_epsilon',
             y='initial_discrep',
             color='blue',
             linestyle='-',
             label='Initial, Planning')
sns.lineplot(data=learning_data,
             ax=ax,
             x='policy_epsilon',
             y='initial_discrep',
             color='blue',
             linestyle='--',
             label='Initial, Learning')
sns.lineplot(data=planning_data,
             ax=ax,
             x='policy_epsilon',
             y='final_discrep',
             color='black',
             linestyle='-',
             label='Final, Planning')
sns.lineplot(data=learning_data,
             ax=ax,
             x='policy_epsilon',
             y='final_discrep',
             color='black',
             linestyle='--',
             label='Final, Learning')

#%%
planning_data = load_analytical_results(planning_up_prob_filename)
learning_data = load_sampled_results(learning_up_prob_filename)

fig, ax = plt.subplots()
plt.plot()
sns.lineplot(data=planning_data,
             ax=ax,
             x='policy_up_prob',
             y='initial_discrep',
             color='blue',
             linestyle='-',
             label='Initial, Planning')
sns.lineplot(data=learning_data,
             ax=ax,
             x='policy_up_prob',
             y='initial_discrep',
             color='blue',
             linestyle='--',
             label='Initial, Learning')
sns.lineplot(data=planning_data,
             ax=ax,
             x='policy_up_prob',
             y='final_discrep',
             color='black',
             linestyle='-',
             label='Final, Planning')
sns.lineplot(data=learning_data,
             ax=ax,
             x='policy_up_prob',
             y='final_discrep',
             color='black',
             linestyle='--',
             label='Final, Learning')
plt.legend()
