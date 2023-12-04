# %% codecell
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from argparse import Namespace
from jax.nn import softmax
from jax.config import config
from pathlib import Path
from collections import namedtuple

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from grl.environment import load_pomdp
from grl.memory.analytical import memory_cross_product
from grl.utils import load_info
from grl.utils.math import greedify
from grl.utils.lambda_discrep import lambda_discrep_measures
from definitions import ROOT_DIR

# %% codecell
# results_dir = Path(ROOT_DIR, 'results', 'final_analytical_kitchen_sinks')
results_dir = Path(ROOT_DIR, 'results', 'final_discrep_kitchen_sinks_pg')

# results_dir = Path(ROOT_DIR, 'results', 'prisoners_dilemma')
# results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_dm')
vi_results_dir = Path(ROOT_DIR, 'results', 'vi')
pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

args_to_keep = ['spec', 'n_mem_states', 'seed']
split_by = [arg for arg in args_to_keep if arg != 'seed']

# this option allows us to compare to either the optimal belief state soln
# or optimal state soln. ('belief' | 'state')
compare_to = 'belief'
# compare_to = 'state'

# policy_optim_alg = 'policy_grad'
policy_optim_alg = 'policy_grad'
# use_memory = 'random_discrete'


# spec_plot_order = [
#     'example_7', 'tmaze_5_two_thirds_up', 'tiger-alt-start', 'paint.95', 'cheese.95', 'network',
#     'shuttle.95', '4x3.95', 'hallway'
# ]
spec_plot_order = [
    'network', 'paint.95', '4x3.95', 'tiger-alt-start', 'shuttle.95', 'cheese.95', 'tmaze_5_two_thirds_up'
]
# spec_plot_order = ['tiger-alt-start','tiger-grid']

# game_name = 'prisoners_dilemma'
# # leader_policies = ['all_d', 'extort', 'tit_for_tat', 'treasure_hunt', 'sugar', 'all_c', 'grudger2', 'alternator', 'majority3']

# leader_policies = ['all_d', 'extort', 'tit_for_tat', 'treasure_hunt', 'sugar', 'grudger2', 'alternator', 'majority3']
# leader_policy_labels = {
#     'all_d': 'all d',
#     'extort': 'extort',
#     'tit_for_tat': 'tit for\ntat',
#     'treasure_hunt': 'treasure\nhunt',
#     'sugar':'sugar',
#     'all_c': 'all c',
#     'grudger2': 'grudger 2',
#     'alternator':'alternator',
#     'majority3':'majority 3'
# }

# spec_plot_order = []
# prisoners_spec_map = {}
# for leader in leader_policies:
#     spec_id = f'{game_name}_{leader}'
#     prisoners_spec_map[spec_id] = leader_policy_labels[leader]
#     spec_plot_order.append(spec_id)

spec_to_belief_state = {'tmaze_5_two_thirds_up': 'tmaze5'}


# %% codecell
compare_to_dict = {}

for spec in spec_plot_order:
    if compare_to == 'belief':

        for fname in pomdp_files_dir.iterdir():
            if 'pomdp-solver-results' in fname.stem:
                if (fname.stem ==
                        f"{spec_to_belief_state.get(spec, spec)}-pomdp-solver-results"
                    ):
                    belief_info = load_info(fname)
                    compare_to_dict[spec] = belief_info['start_val']
                    break
                    # print(f"loaded results for {hparams.spec} from {fname}")
        else:
            for vi_path in vi_results_dir.iterdir():
                for spec in spec_plot_order:
                    if spec_to_belief_state.get(spec, spec) in vi_path.name:
                        vi_info = load_info(vi_path)
                        max_start_vals = vi_info['optimal_vs']
                        compare_to_dict[spec] = np.dot(max_start_vals, vi_info['p0'])

    elif compare_to == 'state':
        for vi_path in vi_results_dir.iterdir():
            if spec_to_belief_state.get(spec, spec) in vi_path.name:
                vi_info = load_info(vi_path)
                max_start_vals = vi_info['optimal_vs']
                compare_to_dict[spec] = np.dot(max_start_vals, vi_info['p0'])

#                 compare_to_list.append(spec_compare_indv)

# %% codecell
compare_to_dict


# %% codecell
all_results = []

for results_path in results_dir.iterdir():
    if results_path.is_dir() or results_path.suffix != '.npy':
        continue

    info = load_info(results_path)

    args = info['args']
    if args['spec'] not in spec_plot_order:
        continue

    if args['policy_optim_alg'] != policy_optim_alg:
        continue

    # if args['use_memory'] != use_memory:
    #     continue

    agent_path = results_path.parent / 'agent' / f'{results_path.stem}.pkl.npy'
    agent = load_info(agent_path)

    pomdp, _ = load_pomdp(args['spec'])
    final_mem_pomdp = memory_cross_product(agent.mem_params, pomdp)

    greedy_policy = greedify(agent.policy)

    greedy_measures = lambda_discrep_measures(final_mem_pomdp, greedy_policy)


    # agent = info['agent']
    init_policy_info = info['logs']['initial_policy_stats']
    init_improvement_info = info['logs']['greedy_initial_improvement_stats']
    final_mem_info = info['logs']['greedy_final_mem_stats']

    def get_perf(info: dict):
        return (info['state_vals_v'] * info['p0']).sum()
    single_res = {k: args[k] for k in args_to_keep}

    # final_mem_perf = get_perf(final_mem_info)
    final_mem_perf = float(get_perf(greedy_measures))

    compare_to_perf = compare_to_dict[args['spec']]
    init_policy_perf = get_perf(init_policy_info)
    init_improvement_perf = get_perf(init_improvement_info)
#     if (final_mem_perf > compare_to_perf):
#         if np.isclose(final_mem_perf, compare_to_perf):
#             final_mem_perf = compare_to_perf
#         else:
#             raise Exception(f"{args['spec']}, compare_to_perf: {compare_to_perf:.3f}, final_mem_perf: {final_mem_perf:.3f}")

    if init_policy_perf > init_improvement_perf:
        init_policy_perf = init_improvement_perf

    single_res.update({
        'init_policy_perf': init_policy_perf,
        'init_improvement_perf': init_improvement_perf,
        'final_mem_perf': final_mem_perf,
        # 'greedy_perf': greedy_perf,
        'compare_to_perf': compare_to_perf,
        # 'init_policy': info['logs']['initial_policy'],
        # 'init_improvement_policy': info['logs']['initial_improvement_policy'],
        # 'final_mem': np.array(agent.memory),
        # 'final_policy': np.array(agent.policy)
    })
    all_results.append(single_res)


all_res_df = pd.DataFrame(all_results)
# %% codecell
all_res_groups = all_res_df.groupby(split_by, as_index=False)
all_res_means = all_res_groups.mean()
del all_res_means['seed']
all_res_means.to_csv(Path(ROOT_DIR, 'results', 'all_pomdps_means.csv'))
# %% codecell
cols_to_normalize = ['init_improvement_perf', 'final_mem_perf']
# merged_df = all_res_df.merge(compare_to_df, on='spec')
merged_df = all_res_df

# for col_name in cols_to_normalize:

normalized_df = merged_df.copy()
normalized_df['init_improvement_perf'] = (normalized_df['init_improvement_perf'] - merged_df['init_policy_perf']) / (merged_df['compare_to_perf'] - merged_df['init_policy_perf'])
normalized_df['final_mem_perf'] = (normalized_df['final_mem_perf'] - merged_df['init_policy_perf']) / (merged_df['compare_to_perf'] - merged_df['init_policy_perf'])
del normalized_df['init_policy_perf']
del normalized_df['compare_to_perf']
# %% codecell
normalized_df.loc[(normalized_df['spec'] == 'hallway') & (normalized_df['n_mem_states'] == 8), 'final_mem_perf'] = 0

# %% codecell
# normalized_df[normalized_df['spec'] == 'prisoners_dilemma_all_c']
seeds = normalized_df[normalized_df['spec'] == normalized_df['spec'][0]]['seed'].unique()
# %% codecell
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

#     spec_map |= prisoners_spec_map

    if id not in spec_map:
        return id
    return spec_map[id]

groups = normalized_df.groupby(split_by, as_index=False)
means = groups.mean()
means['init_improvement_perf'].clip(lower=0, upper=1, inplace=True)
means['final_mem_perf'].clip(lower=0, upper=1, inplace=True)

std_errs = groups.std()
std_errs['init_improvement_perf'] /= np.sqrt(len(seeds))
std_errs['final_mem_perf'] /= np.sqrt(len(seeds))

# SORTING
sorted_mean_df = pd.DataFrame()
sorted_std_err_df = pd.DataFrame()

for spec in spec_plot_order:
    mean_spec_df = means[means['spec'] == spec]
    std_err_spec_df = std_errs[std_errs['spec'] == spec]
    sorted_mean_df = pd.concat([sorted_mean_df, mean_spec_df])
    sorted_std_err_df = pd.concat([sorted_std_err_df, std_err_spec_df])

means = sorted_mean_df
std_errs = sorted_std_err_df

num_n_mem = list(sorted(normalized_df['n_mem_states'].unique()))

group_width = 1
bar_width = group_width / (len(num_n_mem) + 2)
fig, ax = plt.subplots(figsize=(12, 6))

specs = means[means['n_mem_states'] == num_n_mem[0]]['spec']
# spec_order_mapping = [spec_plot_order.index(s) for s in specs]
spec_order_mapping = np.arange(len(specs), dtype=int)

xlabels = [maybe_spec_map(l) for l in specs]
x = np.arange(len(specs))

init_improvement_perf_mean = np.array(means[means['n_mem_states'] == num_n_mem[0]]['init_improvement_perf'])
init_improvement_perf_std = np.array(std_errs[std_errs['n_mem_states'] == num_n_mem[0]]['init_improvement_perf'])

ax.bar(x + (0 + 1) * bar_width,
       init_improvement_perf_mean,
       bar_width,
       yerr=init_improvement_perf_std,
       label='Memoryless',
       color='#5B97E0')

mem_colors = ['#E0B625', '#DD8453', '#C44E52']

for i, n_mem_states in enumerate(num_n_mem):
    curr_mem_mean = np.array(means[means['n_mem_states'] == n_mem_states]['final_mem_perf'])
    curr_mem_std = np.array(std_errs[std_errs['n_mem_states'] == n_mem_states]['final_mem_perf'])
    ax.bar(x + (i + 2) * bar_width,
           curr_mem_mean,
           bar_width,
           yerr=curr_mem_std,
           label=f"{int(np.log2(n_mem_states))} Memory Bits",
           color=mem_colors[i])

ax.set_ylim([0, 1.05])
ax.set_ylabel(f'Relative Performance\n (w.r.t. optimal {compare_to} & initial policy)')
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels)
# ax.legend(bbox_to_anchor=(0.317, 0.62), framexalpha=0.95)
# ax.set_title(f"Memory Iteration ({policy_optim_alg})")
ax.set_title(f"Memory: ({policy_optim_alg})")

downloads = Path().home() / 'Downloads'
fig_path = downloads / f"{results_dir.stem}.pdf"
fig.savefig(fig_path, bbox_inches='tight')
# %% codecell
