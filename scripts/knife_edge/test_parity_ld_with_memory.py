from jax import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from grl.environment import load_pomdp
from grl.memory.lib import get_memory
from grl.utils.loss import mem_discrep_loss

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
np.set_printoptions(precision=8)

spec = 'parity_check'
seed = 2020
n_samples = 10000

rng = random.PRNGKey(seed=seed)
np.random.seed(seed)

pomdp, info = load_pomdp(spec)
pi = info['Pi_phi'][0]

lds = []
for i in range(n_samples):
    mem_params = get_memory('0', pomdp.observation_space.n, pomdp.action_space.n,
                            n_mem_states=2)

    # TODO: maybe also randomly sample?
    inp_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    mem_ld = mem_discrep_loss(mem_params, inp_aug_pi, pomdp).item()
    lds.append(mem_ld)

def plot_loghist(x, bins):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(x, bins=logbins)
  plt.xscale('log')

#%%
plt.subplots(figsize=(7,2.5))
plot_loghist(lds, 100)
plt.title('Parity Check with Random Memory')
plt.xlabel(r'$\lambda$-discrepancy ($\Lambda$)')
plt.ylabel('Number of memory functions')
plt.tight_layout()
plt.savefig('parity-check-memory.png')
plt.show()
