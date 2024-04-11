from collections import namedtuple
import glob
import json
import os
from pathlib import Path

import jax.numpy as jnp
from jax.nn import softmax
from jax.config import config
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

from definitions import ROOT_DIR
from grl.environment import load_spec
from grl.mdp import MDP, POMDP
from grl.utils import load_info
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.discrete_search import generate_hold_mem_fn
from grl.memory import memory_cross_product

plot_dir = 'results/plots/iclr2023/'
os.makedirs(plot_dir, exist_ok=True)
policy_optim_alg = 'policy_iter'

title_note = 'PI' if policy_optim_alg == 'policy_iter' else 'PG'

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

# %%
# results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_pi')
# results_dir = Path(ROOT_DIR, 'results', 'all_pomdps_mi_pi_obs_space')
# results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_dm')
# results_dir = Path(ROOT_DIR, 'results', 'final_analytical') # analytical-optimized mem fn
results_dir = Path(ROOT_DIR, 'results', 'final_analytical_kitchen_sinks') # analytical-optimized mem fn
# results_dir = Path(ROOT_DIR, 'results', 'random_discrete_analytical') # determ. random mem fn
# results_dir = Path(ROOT_DIR, 'results', 'random_uniform_analytical') # stoch. random mem fn
vi_results_dir = Path(ROOT_DIR, 'results', 'vi')
pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

args_to_keep = ['spec', 'n_mem_states', 'seed']
split_by = [arg for arg in args_to_keep if arg != 'seed']

# this option allows us to compare to either the optimal belief state soln
# or optimal state soln. ('belief' | 'state')
compare_to = 'belief'

# spec_plot_order = ['example_7', 'slippery_tmaze_5_two_thirds_up',
#                    'tiger', 'paint.95', 'cheese.95',
#                    'network', 'shuttle.95', '4x3.95']
# spec_plot_order = [
#     'example_7', 'tmaze_5_two_thirds_up', 'tiger-alt-start', 'paint.95', 'cheese.95', 'network',
#     'shuttle.95', '4x3.95', 'hallway'
# ]
spec_plot_order = [
    'paint.95',
    'tiger-alt-start',
    'shuttle.95',
    'network',
    '4x3.95',
    'cheese.95',
    'tmaze_5_two_thirds_up',
    # 'example_7',
]

spec_to_belief_state = {'tmaze_5_two_thirds_up': 'tmaze5'}
calibrations_data = pd.read_csv('results/discrete/all_pomdps_means_fixed.csv').query('n_mem_states==2').set_index('spec')
calibrations_dict = calibrations_data.to_dict('index')
# calibrations_data = calibrations_data.reset_index()
# scale_low = calibrations_data['init_policy_perf']
# scale_high = calibrations_data['compare_to_perf']
# scaled_final_values = (calibrations_data['final_mem_perf'] - scale_low) / (scale_high - scale_low)
# calibrations_data['scaled_final_value'] = scaled_final_values
# calibrations_data['env'] = calibrations_data['spec'].map(maybe_spec_map)

# %%

compare_to_list = []

# if compare_to == 'belief':
for spec in spec_plot_order:
    for fname in pomdp_files_dir.iterdir():
        if 'pomdp-solver-results' in fname.stem:
            if (fname.stem == f"{spec_to_belief_state.get(spec, spec)}-pomdp-solver-results"):
                belief_info = load_info(fname)
                coeffs = belief_info['coeffs']
                max_start_vals = coeffs[belief_info['max_start_idx']]
                spec_compare_indv = {
                    'spec': spec,
                    'compare_perf': np.dot(max_start_vals, belief_info['p0'])
                }
                compare_to_list.append(spec_compare_indv)
                break
                # print(f"loaded results for {hparams.spec} from {fname}")
    else:
        for vi_path in vi_results_dir.iterdir():
            for spec in spec_plot_order:
                if spec_to_belief_state.get(spec, spec) in vi_path.name:
                    vi_info = load_info(vi_path)
                    spec_compare_indv = {
                        'spec': spec,
                        'compare_perf': np.dot(max_start_vals, belief_info['p0'])
                    }
                    compare_to_list.append(spec_compare_indv)

# elif compare_to == 'state':
# else:
#     raise NotImplementedError

compare_to_df = pd.DataFrame(compare_to_list)

# %%

all_results = []

for results_path in results_dir.iterdir():
    if results_path.is_dir() or results_path.suffix != '.npy':
        continue
    info = load_info(results_path)

    args = info['args']

    if args['policy_optim_alg'] != policy_optim_alg:
        continue

    # agent = info['agent']
    init_policy_info = info['logs']['initial_policy_stats']
    init_improvement_info = info['logs']['greedy_initial_improvement_stats']
    final_mem_info = info['logs']['greedy_final_mem_stats']

    def get_perf(info: dict):
        return (info['state_vals_v'] * info['p0']).sum()

    single_res = {k: args[k] for k in args_to_keep}

    single_res.update({
        'init_policy_perf': get_perf(init_policy_info),
        'init_improvement_perf': get_perf(init_improvement_info),
        'final_mem_perf': get_perf(final_mem_info),
        # 'init_policy': info['logs']['initial_policy'],
        # 'init_improvement_policy': info['logs']['initial_improvement_policy'],
        # 'final_mem': np.array(agent.memory),
        # 'final_policy': np.array(agent.policy)
    })
    all_results.append(single_res)

all_res_df = pd.DataFrame(all_results)

# %%
cols_to_normalize = ['init_improvement_perf', 'final_mem_perf']
merged_df = compare_to_df.merge(all_res_df, on='spec')

# for col_name in cols_to_normalize:

normalized_df = merged_df.copy()
for env in spec_plot_order:
    scales = calibrations_dict[env]
    idx = normalized_df['spec'] == env
    normalized_df.loc[idx, 'init_improvement_perf'] = (
        normalized_df[idx]['init_improvement_perf'] -
        scales['init_policy_perf']) / (scales['compare_to_perf'] - scales['init_policy_perf'])
    normalized_df.loc[idx, 'final_mem_perf'] = (
        normalized_df[idx]['final_mem_perf'] -
        scales['init_policy_perf']) / (scales['compare_to_perf'] - scales['init_policy_perf'])

# del normalized_df['init_policy_perf']
# del normalized_df['compare_perf']

# all_normalized_perf_results = {}
# for hparams, res in all_results.items():
#     max_key = 'compare_perf'
#     # if max_key not in res:
#     #     max_key = 'final_mem_perf'
#     max_v = res[max_key]
#     min_v = res['init_policy_perf']
#     for k, v in res.items():
#         if '_perf' in k:
#             all_results[hparams][k] = (v - min_v) / (max_v - min_v)
# %%
normalized_df.groupby(split_by).mean()

# %%

# all_table_results = {}
# all_plot_results = {'x': [], 'xlabels': []}
#
# for i, spec in enumerate(spec_plot_order):
#     hparams = sorted([k for k in all_results.keys() if k.spec == spec],
#                      key=lambda hp: hp.n_mem_states)
#
#     first_res = all_results[hparams[0]]
#     all_plot_results['x'].append(i)
#     all_plot_results['xlabels'].append(spec)
#
#     # we first add initial and first improvement stats
#     for k, v in first_res.items():
#         if 'perf' in k and k != 'final_mem_perf':
#             mean = v.mean(axis=0)
#             std_err = v.std(axis=0) / np.sqrt(v.shape[0])
#
#             stripped_str = k.replace('_perf', '')
#             if stripped_str not in all_plot_results:
#                 all_plot_results[stripped_str] = {'mean': [], 'std_err': []}
#             all_plot_results[stripped_str]['mean'].append(mean)
#             all_plot_results[stripped_str]['std_err'].append(std_err)
#
#     # now we add final memory perf, for each n_mem_states
#     for hparam in hparams:
#         res = all_results[hparam]
#         for k, v in res.items():
#             if k == 'final_mem_perf':
#                 mean = v.mean(axis=0)
#                 std_err = v.std(axis=0) / np.sqrt(v.shape[0])
#                 stripped_str = k.replace('_perf', '')
#                 mem_label = f"mem, |ms| = {hparam.n_mem_states}"
#                 if mem_label not in all_plot_results:
#                     all_plot_results[mem_label] = {'mean': [], 'std_err': []}
#                 all_plot_results[mem_label]['mean'].append(mean)
#                 all_plot_results[mem_label]['std_err'].append(std_err)
#
# ordered_plot = []
# # ordered_plot.append(('init_policy', all_plot_results['init_policy']))
# ordered_plot.append(('init_improvement', all_plot_results['init_improvement']))
# for k in sorted(all_plot_results.keys()):
#     if 'mem' in k:
#         ordered_plot.append((k, all_plot_results[k]))
# ordered_plot.append(('state_optimal', all_plot_results['vi']))

# %%

# %%
def maybe_spec_map(id: str):
    spec_map = {
        '4x3.95': '4x3',
        'cheese.95': 'cheese',
        'paint.95': 'paint',
        'shuttle.95': 'shuttle',
        'example_7': 'ex. 7',
        'tmaze_5_two_thirds_up': 'tmaze',
        'tiger-alt-start': 'tiger'
    }
    if id not in spec_map:
        return id
    return spec_map[id]

groups = normalized_df.groupby(split_by, sort=False, as_index=False)
means = groups.mean()
std_errs = groups.sem()
num_n_mem = list(sorted(normalized_df['n_mem_states'].unique()))

group_width = 1
bar_width = group_width / (len(num_n_mem) + 2)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(means['spec'].unique()))
xlabels = [maybe_spec_map(l) for l in list(means['spec'])]

ax.bar(x[:len(means) // 3] + (0 + 1) * bar_width,
       means[means['n_mem_states'] == num_n_mem[0]]['init_improvement_perf'],
       bar_width,
       yerr=std_errs[std_errs['n_mem_states'] == num_n_mem[0]]['init_improvement_perf'],
       label='Memoryless',
       color='#5B97E0')
bar_colors = ['xkcd:goldenrod', 'tab:orange', '#E05B5D']
bar_colors = ['#E0B625', '#DD8453', '#C44E52']
# bar_colors = ['#', '#E05B5D', 'tab:orange']

for i, n_mem_states in enumerate(num_n_mem):
    ax.bar(x + (i + 2) * bar_width,
           means[means['n_mem_states'] == n_mem_states]['final_mem_perf'],
           bar_width,
           yerr=std_errs[std_errs['n_mem_states'] == n_mem_states]['final_mem_perf'],
           label=f"{int(np.log(n_mem_states))+1} Bit"+('s' if i > 0 else ''),
           color=bar_colors[i])

handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
ax.legend(handles, labels, loc='upper center', framealpha=0.8, ncols=4, bbox_to_anchor=(0.45, -.1))

ax.set_ylabel(f'Normalized Return\n (0 = random, 1 = belief-states)')

ax.set_title(f"Gradient-Based Memory Optimization ({title_note})")

ax.set_ylim([0, 1.05])
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels[::3])
ax.hlines(1, x.min(), x.max() + 1, ls='--', color='k', alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

fig_path = plot_dir + f'/analytical_{policy_optim_alg}.pdf'
fig.savefig(fig_path)

#%%

def load_results(pathname):
    all_results = []
    results_dirs = glob.glob(pathname)
    for results_dir in results_dirs:
        results_file = results_dir + '/discrete_oracle.json'
        with open(results_file, 'r') as f:
            info = json.load(f)
            if info['env'] not in spec_plot_order:
                continue
            trial_id = int(info['trial_id'].split('_')[1]) % 10
            info['trial_id'] = trial_id
            scales = calibrations_dict[info['env']]
            info['final_mem_perf'] = (info['end_value'] - scales['init_policy_perf']) / (
                scales['compare_to_perf'] - scales['init_policy_perf'])
            if info['tmax'] < info['tmin']:
                continue
            del info['optimizer_info']
            all_results.append(info)
    data = pd.DataFrame(all_results)
    return data

# data = load_results('results/discrete/tune07-1repeats*/*/*')
discrete_oracle_data = load_results('results/discrete/locality07*/*/*')
discrete_oracle_data['spec'] = discrete_oracle_data['env'] #.map(maybe_spec_map)
del discrete_oracle_data['env']
discrete_oracle_data = discrete_oracle_data.query(f'spec in {spec_plot_order}')
# discrete_oracle_data['n_mem_states'] = 1
# del discrete_oracle_data['study_name']
# del discrete_oracle_data['mem_optimizer']
# del discrete_oracle_data['policy_optimization']

discrete_oracle_data['spec'] = discrete_oracle_data['spec'].sort_values()
split_by = ['spec', 'n_mem_states', 'policy_optim_alg', 'mem_optimizer', 'init_policy_randomly']
group = discrete_oracle_data.groupby(split_by, sort=False, as_index=False)

def sort_specs(series):
    return pd.Series([spec_plot_order.index(x) for x in series])

discrete_oracle_means = group.mean(numeric_only=True).sort_values(by='spec',
                                                                  key=sort_specs,
                                                                  ignore_index=True)
discrete_oracle_std_errs = group.sem(numeric_only=True).sort_values(by='spec',
                                                                    key=sort_specs,
                                                                    ignore_index=True)

means_with_discrete = pd.concat([means, discrete_oracle_means])
std_errs_with_discrete = pd.concat([std_errs, discrete_oracle_std_errs])

means_with_discrete['policy_optim_alg'].fillna(policy_optim_alg, inplace=True)
means_with_discrete['mem_optimizer'].fillna('analytical', inplace=True)
means_with_discrete['init_policy_randomly'].fillna(False, inplace=True)
std_errs_with_discrete['policy_optim_alg'].fillna(policy_optim_alg, inplace=True)
std_errs_with_discrete['mem_optimizer'].fillna('analytical', inplace=True)
std_errs_with_discrete['init_policy_randomly'].fillna(False, inplace=True)

# sns.barplot(data=normalized_df, x='spec', y='final_mem_perf', hue='n_mem_states')
# plt.tight_layout()
# plt.xticks(rotation=90)
# plt.show()

#%%
x = np.arange(len(means['spec'].unique()))
num_n_mem = list(sorted(means_with_discrete['n_mem_states'].unique()))
xlabels = [maybe_spec_map(l) for l in list(spec_plot_order)]

settings_list = [
    ('analytical', ''),
    ('annealing', '\\\\'),
]

subset = f'policy_optim_alg == "{policy_optim_alg}"'

unique_runs = sorted(
    pd.unique(
        list(map(str,
                 means_with_discrete.query(subset)[['n_mem_states', 'mem_optimizer']].values))))
n_bars = len(unique_runs) + 1
bar_width = 1 / (n_bars + 2)

mpl.rcParams['hatch.color'] = 'k'
fig, ax = plt.subplots(figsize=(12, 6))
query = 'n_mem_states == 2 and mem_optimizer == "analytical"'
ax.bar(x + (0 + 1) * bar_width,
       means_with_discrete.query(subset).query(query)['init_improvement_perf'],
       bar_width,
       yerr=std_errs_with_discrete.query(subset).query(query)['init_improvement_perf'],
       label='Memoryless',
       color='#5B97E0')
# bar_colors = ['xkcd:goldenrod', 'tab:orange', '#E05B5D']
bar_colors = ['#E0B625', '#DD8453', '#C44E52']
# bar_colors = ['#', '#E05B5D', 'tab:orange']

# settings_list = [
#     ('annealing', 'none', '+'),
#     ('annealing', 'td', 'X'),
#     ('analytical', 'td', ''),
# ]

for chunk, (mem_optimizer, hatching) in enumerate(settings_list):
    for i, n_mem_states in enumerate(num_n_mem):
        query = (f'n_mem_states == {n_mem_states} '
                 f'and mem_optimizer == "{mem_optimizer}" '
                 f'and policy_optim_alg == "{policy_optim_alg}"')
        optim_name = {'annealing': 'HC', 'analytical': 'Grad'}[mem_optimizer]
        try:
            plt.bar(x + (3 * chunk + i + 2) * bar_width,
                    means_with_discrete.query(query)['final_mem_perf'],
                    bar_width,
                    yerr=std_errs_with_discrete.query(query)['final_mem_perf'],
                    label=f"{int(np.log(n_mem_states))+1}-bit, {optim_name}",
                    color=bar_colors[i],
                    hatch=hatching)
        except ValueError as e:
            x_alt = np.array([0, 1, 2, 3, 4, 7])
            plt.bar(x_alt + (3 * chunk + i + 2) * bar_width,
                    means_with_discrete.query(query)['final_mem_perf'],
                    bar_width,
                    yerr=std_errs_with_discrete.query(query)['final_mem_perf'],
                    label=f"{int(np.log(n_mem_states))+1}-bit, {optim_name}",
                    color=bar_colors[i],
                    hatch=hatching)

handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
empty_patch = mpatches.Patch(color='none')
handles.insert(4, empty_patch)
labels.insert(4, '')
first_handles, last_handles = handles[:len(handles)//2], handles[len(handles)//2:]
first_labels, last_labels = labels[:len(labels)//2], labels[len(labels)//2:]
handles = [val for tup in zip(*[first_handles, last_handles]) for val in tup]
labels = [val for tup in zip(*[first_labels, last_labels]) for val in tup]
ax.legend(handles, labels, loc='upper center', framealpha=0.8, ncols=4, bbox_to_anchor=(0.5, -.1))

ax.set_ylim([0, 1.05])
ax.set_ylabel(f'Normalized Return')
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels)
# ax.legend(
#     loc='upper left',
#     framealpha=0.8,
#     ncols=2,
# )
ax.set_title(f"Performance with Memory Optimization")
ax.hlines(1, x.min(), x.max() + 1, ls='--', color='k', alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

fig_path = plot_dir + f'annealing_{policy_optim_alg}.pdf'
fig.savefig(fig_path)
fig.show()
