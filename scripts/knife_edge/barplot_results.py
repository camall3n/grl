import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('results_batch_run_kitchen.csv')
belief = data.query('objective == "belief"').groupby(['spec'])['score'].mean()
random = data.query('objective == "random"').groupby(['spec'])['score'].mean()
utility_range = belief - random
utility_base = random

data['spec'].unique()

data = data.join(utility_base, on='spec', rsuffix='_base')
data = data.join(utility_range, on='spec', rsuffix='_range')
data['Normalized Return'] = (data['score'] - data['score_base']) / data['score_range']

data['n_mem_bits'] = np.log2(data['n_mem_states'])
data['n_mem_bits'] = data['n_mem_bits'].fillna(0).astype(int)

env_names = {
    'paint.95': 'paint',
    'tmaze_5_two_thirds_up': 't-maze',
    '4x3.95': '4x3',
    'cheese.95': 'cheese',
    'tiger-alt-start': 'tiger',
    'shuttle.95': 'shuttle',
    'network': 'network',
}
col_names = {
    'spec': 'Environment',
    'n_mem_bits': 'Memory Size (bits)',
}
data = data.replace(to_replace=env_names)
data = data.rename(columns=col_names)

fig, ax = plt.subplots(figsize=(8, 3))
sns.barplot(data=data, x='Environment', y='Normalized Return', hue='Memory Size (bits)', ax=ax)
handles = ax.get_legend().legend_handles
texts = [x.get_text() for x in ax.get_legend().texts]
title = ax.get_legend().get_title().get_text()

ax.legend(title=title, handles=handles, labels=texts, ncols=4, loc='lower right', fancybox=True, framealpha=1)
xlims = ax.get_xlim()
ax.hlines(1, *xlims, color='k', ls='--', alpha=0.5)


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
# ax.set_ylabel(f'Normalized Return')
# ax.set_xticks(x + group_width / 2)
# ax.set_xticklabels(xlabels)
# ax.legend(
#     loc='upper left',
#     framealpha=0.8,
#     ncols=2,
# )
ax.set_title(f"Performance with Memory Optimization")

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
