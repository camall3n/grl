import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from jax.config import config
from pathlib import Path

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

#%%

discrete_data = []
for filename in glob.glob('results/discrete/discrete05*/*/*/discrete_oracle.json'):
    with open(filename, 'r') as f:
        results = json.load(f)
        results['spec'] = filename.split('/')[3]
        results['seed'] = int(filename.split('/')[4])
        discrete_data.append(results)

discrete_df = pd.DataFrame(discrete_data).groupby('spec').mean()
discrete_df['end_value_std'] = (
    pd.DataFrame(discrete_data).groupby('spec').std()['end_value'] /
    np.sqrt(pd.DataFrame(discrete_data).groupby('spec').count()['seed']))

# %%
compare_to = 'belief'

calibration_df = pd.read_csv('results/discrete/all_pomdps_means.csv')
merged_df = calibration_df.merge(discrete_df, how='inner', on='spec')

normalized_df = merged_df.copy()

def normalize(data, col, ref_data, low_col, high_col):
    data[col] = (data[col] - ref_data[low_col]) / (ref_data[high_col] - ref_data[low_col])

normalize(normalized_df, 'init_improvement_perf', merged_df, 'init_policy_perf', 'compare_to_perf')
normalize(normalized_df, 'final_mem_perf', merged_df, 'init_policy_perf', 'compare_to_perf')
normalize(normalized_df, 'end_value', merged_df, 'init_policy_perf', 'compare_to_perf')

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

# groups = normalized_df.groupby('spec', as_index=False)
means = normalized_df
# std_errs = groups.std()
num_n_mem = list(sorted(normalized_df['n_mem_states'].unique()))

group_width = 1
bar_width = group_width / (len(num_n_mem) + 2)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(means))
xlabels = [maybe_spec_map(l) for l in list(means['spec'])]

ax.bar(
    x + (0 + 1) * bar_width,
    means[means['n_mem_states'] == num_n_mem[0]]['init_improvement_perf'],
    bar_width,
    #    yerr=std_errs[std_errs['n_mem_states'] == num_n_mem[0]]['init_improvement_perf'],
    label='Memoryless')

for i, n_mem_states in enumerate(num_n_mem):
    ax.bar(
        x + (i + 2) * bar_width,
        # means[means['n_mem_states'] == n_mem_states]['final_mem_perf'],
        means[means['n_mem_states'] == n_mem_states]['end_value'],
        bar_width,
        # yerr=std_errs[std_errs['n_mem_states'] == n_mem_states]['final_mem_perf'],
        # yerr=discrete_df['end_value_std'],
        label=f"{int(np.log2(n_mem_states))} Memory Bits")
ax.set_ylim([0, 1])
ax.set_ylabel(f'Relative Performance\n (w.r.t. optimal {compare_to} & initial policy)')
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels)
ax.legend(bbox_to_anchor=(0.7, 0.6), framealpha=0.95)
ax.set_title("Performance of Memory Iteration in POMDPs")

downloads = Path().home() / 'Downloads'
fig_path = downloads / f"no-prio.pdf"
fig.savefig(fig_path)
