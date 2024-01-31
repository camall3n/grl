# %% codecell
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from argparse import Namespace
from jax.nn import softmax
from jax import config
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from scripts.plotting.parse_experiments import parse_baselines, parse_dirs
from definitions import ROOT_DIR

# %% codecell
# results_dir = Path(ROOT_DIR, 'results', 'final_analytical_kitchen_sinks')
experiment_dirs = [
    Path(ROOT_DIR, 'results', 'mem_tde_kitchen_sinks_pg'),
    Path(ROOT_DIR, 'results', 'final_discrep_kitchen_sinks_pg'),
]

# results_dir = Path(ROOT_DIR, 'results', 'prisoners_dilemma')
# results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_dm')
vi_results_dir = Path(ROOT_DIR, 'results', 'vi')
pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

args_to_keep = ['spec', 'n_mem_states', 'seed', 'alpha', 'residual']
split_by = [arg for arg in args_to_keep if arg != 'seed'] + ['experiment']

# this option allows us to compare to either the optimal belief state soln
# or optimal state soln. ('belief' | 'state')
compare_to = 'belief'
# compare_to = 'state'

# policy_optim_alg = 'policy_grad'
policy_optim_alg = 'policy_grad'

spec_plot_order = [
    'network', 'paint.95', '4x3.95', 'tiger-alt-start', 'shuttle.95', 'cheese.95', 'tmaze_5_two_thirds_up'
]

# %% codecell

compare_to_dict = parse_baselines(spec_plot_order,
                                  vi_results_dir,
                                  pomdp_files_dir,
                                  compare_to=compare_to)


# %% codecell
all_res_df = parse_dirs(experiment_dirs,
                        compare_to_dict,
                        args_to_keep)



# %% codecell

# FILTER OUT for what we want to plot
# alpha = 1.
#
# residual = True
# filtered_df = all_res_df[(all_res_df['alpha'] == alpha) & (all_res_df['residual'] == residual)].reset_index()

# %% codecell
all_res_groups = all_res_df.groupby(split_by, as_index=False)
all_res_means = all_res_groups.mean()
del all_res_means['seed']
# all_res_means.to_csv(Path(ROOT_DIR, 'results', 'all_pomdps_means.csv'))

# %% codecell
cols_to_normalize = ['init_improvement_perf', 'final_mem_perf']
# merged_df = filtered_df.merge(compare_to_df, on='spec')
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
alpha_str = 'uniform' if alpha == 1. else 'occupancy'
residual_str = 'semi_grad' if not residual else 'residual'
ax.set_title(f"Memory: (MSTDE w/ {policy_optim_alg} ({residual_str}, {alpha_str}))")

downloads = Path().home() / 'Downloads'
fig_path = downloads / f"{results_dir.stem}_{residual_str}_{alpha_str}.pdf"
fig.savefig(fig_path, bbox_inches='tight')
# %% codecell
