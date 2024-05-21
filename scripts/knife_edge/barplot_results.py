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

ax.legend(title=title, handles=handles, labels=texts, ncols=4, loc='upper left', fancybox=True)
xlims = ax.get_xlim()
ax.hlines(1, *xlims, color='k', ls='--', alpha=0.5)

ax.set_ylim([0.3,1.1])
ax.set_title(f"Performance with Memory Optimization")


ax_lim = ax.get_ylim()
ax_rng = ax_lim[1] - ax_lim[0]

d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d/ax_rng, +d/ax_rng), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d/ax_rng, +d/ax_rng), **kwargs)  # top-right diagonal

top_y=0.08
bottom_y=0.05

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (top_y - d/ax_rng, top_y + d/ax_rng), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (top_y - d/ax_rng, top_y + d/ax_rng), **kwargs)  # bottom-right diagonal
ax.plot((-d, +d), (bottom_y - d/ax_rng, bottom_y + d/ax_rng), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (bottom_y - d/ax_rng, bottom_y + d/ax_rng), **kwargs)  # bottom-right diagonal

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig('nice-barplot.png')
plt.show()

data['seed'].unique()
