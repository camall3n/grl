import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans serif"],
    "font.monospace": ["Computer Modern Typewriter"],
    "axes.labelsize": 12,  # LaTeX default is 10pt
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

hex = [
    '#ff96b6',# pink
    '#df5b5d',# red
    '#DD8453',# orange
    '#f8de7c',# yellow
    '#3FC57F',# green
    '#48dbe5',# cyan
    '#3180df',# blue
    '#9d79cf',# purple
    '#886a2c',# brown
    '#ffffff',# white
    '#d5d5d5',# light gray
    '#666666',# dark gray
    '#000000',# black
]
p = sns.color_palette(hex, as_cmap=False)
p

hex = [
    '#3FC57F',#g
    '#3180df',#b
    # '#df5b5d',#r
    # '#9d79cf',#purp
    # '#f8de7c',#y
    # '#ff96b6',#pink
    # '#48dbe5',#teal
    # '#886a2c',#brown
    '#666666',#gray
    # '#DD8453',#o
]
palette = sns.color_palette(hex)
palette
col_wrap = None
# col_wrap = 2
# test = True
test = False

data = pd.read_csv('results_rnn_30seeds.csv')
data['seed'].unique()[0]
g = sns.relplot(
    data=data.query('seed==0') if test else data,
    x='timestep',
    y='score',
    hue='algo',
    col='env',
    col_wrap=col_wrap,
    kind='line',
    facet_kws={'sharey': False, 'sharex': False},
    palette=palette,
    height=2.5 if col_wrap==2 else 3,
    aspect=1.4 if col_wrap==2 else 1,
)

# Clean up the titles
axes = g.axes.flatten()
titles = [
    r'RockSample $11\times 11$',
    r'RockSample $15\times 15$',
    r'Battleship $10\times 10$',
    r'P.O. PacMan'
]
for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('')

axes[0].set_ylabel('Discounted Return')
axes[2].set_ylabel('Discounted Return')

# Clean up the legend
leg = g._legend
leg.set_bbox_to_anchor([0.1,0.8])
handles = leg.legend_handles
texts = [x.get_text() for x in leg.texts]
nice_algo_names = {
    'ld_ppo': r'PPO+RNN+LD',
    'ppo': r'PPO+RNN',
    'memoryless_ppo': 'PPO',
}
texts = [nice_algo_names[text] for text in texts]
title = leg.get_title().get_text()
leg.remove()
axes[-1].legend(handles=handles, labels=texts, loc='lower right', fancybox=True, framealpha=0.7)

plt.tight_layout()
if not test:
    plt.savefig(f'nice-rnn{"" if col_wrap is None else "-2x2"}.png')
# plt.savefig('nice-rnn.png')
plt.show()
