import glob
import os

import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from numpy.linalg import norm
import optuna
import pandas as pd
import seaborn as sns

from grl import environment
from grl.mdp import POMDP, MDP
from grl.agent.actorcritic import ActorCritic
from grl.memory.lib import get_memory

#%%
summary_filepath = 'results/sample_based/exp12-tmaze_2_two_thirds_up-summary.pkl'
data = pd.read_pickle(summary_filepath)
samples = np.concatenate(
    (np.array(data['mem_coords'].tolist()), data['value'].to_numpy().reshape(-1, 1)), axis=-1)
df_sampled = pd.DataFrame(data=samples, columns=['y', 'p1', 'p2', 'D'])

#%%
summary_filepath = 'results/analytical_tmaze_plot_data.pkl'
df_analytical = pd.read_pickle(summary_filepath)

#%%
def plot_optimization_surface(df, y_range=[0, 0.05], ax=None):
    vmin = df['D'].min()
    vmax = df['D'].max()
    frame_data = df.query(f"{y_range[0]} <= y <= {y_range[1]}")
    x = frame_data['p1'].to_numpy()
    y = frame_data['p2'].to_numpy()
    z = frame_data['D'].to_numpy()
    if ax is None:
        fig, ax = plt.subplots()
    ax.tricontourf(x, y, z, levels=np.linspace(vmin, vmax, 10), vmin=vmin, vmax=vmax)
    ax.tick_params(axis='both', which='both', direction='in', labelcolor='w')
    ax.set_xlabel("$p1$")
    ax.set_ylabel("$p2$")
    ax.set_title(f"$y \in [{','.join(map(str, np.array(y_range).round(2)))}]$")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('square')

def plot_surface_slices(df):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))
    y_bins = np.linspace(0, 1, 10)
    for ax, ymin, ymax in zip(axes.flatten(), y_bins[:-1], y_bins[1:]):
        try:
            plot_optimization_surface(df, y_range=[ymin, ymax], ax=ax)
        except:
            continue
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    cax.set_title('$D_\lambda$')
    cax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=mpl.cm.viridis, orientation='vertical')
    plt.gcf().add_axes(cax)
    plt.savefig('foo.png')

#%%
plot_surface_slices(df_sampled)

#%%
df = df_analytical
p1 = df.p1.to_numpy().reshape(100, 100, -1)
p2 = df.p2.to_numpy().reshape(100, 100, -1)
D = df.D.to_numpy().reshape(100, 100, -1)
y = df.y.to_numpy().reshape(100, 100, -1)

#%%
frames = []
for i in range(D.shape[-1]):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.contourf(p1[:, :, 0], p2[:, :, 0], D[:, :, i], levels=np.linspace(vmin, vmax, 20))
    canvas.draw() # draw the canvas, cache the renderer
    # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    frames.append(image)
imageio.mimwrite('foo.mp4', frames)

#%%
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
z = 3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) - 10 * (
    x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2) - 1 / 3 * np.exp(-(x + 1)**2 - y**2)
plt.contourf(x, y, z, levels=20)

sns.kdeplot(data=df, x='y', clip=[0, 1])
