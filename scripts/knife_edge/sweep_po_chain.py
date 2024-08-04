
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import jax
from jax.config import config

from grl.environment import load_pomdp
from grl.utils.loss import discrep_loss
from grl.utils.policy_eval import analytical_pe

#%%

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

spec = 'simple_chain'
seed = 42

np.set_printoptions(precision=8, suppress=True)
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

rand_key = None
np.random.seed(seed)
rand_key = jax.random.PRNGKey(seed)

pomdp, pi_dict = load_pomdp(spec, rand_key)

if 'Pi_phi' in pi_dict and pi_dict['Pi_phi'] is not None:
    pi = pi_dict['Pi_phi'][0]
    # print(f'Pi_phi:\n {pi_phi}')
    pi_phi = pomdp.get_ground_policy(pi)
    pi_phi = np.ones((11,1), dtype=float)

phi = pomdp.phi
mdp_phi = np.concatenate([np.eye(10), np.zeros((10,1))], axis=-1)

pomdp_phi_single = np.array([
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
#%%
po_types = {
    'single': pomdp_phi_single,
}

lds = []
gammas = [0.95, 0.9]
ps = np.linspace(0, 1, 500)
for p in tqdm(ps):
    for po_type, phi in po_types.items():
        for gamma in gammas:
            pomdp.phi = p * phi + (1-p) * mdp_phi
            pi = pi_phi
            pomdp.gamma = gamma
            state_vals, mc_vals, td_vals, info = analytical_pe(pi, pomdp)
            lds.append({'p': p, 'ld': discrep_loss(pi, pomdp, alpha=0, error_type='max')[0].item(), 'po_type': po_type, 'gamma': gamma})

    # pomdp.phi = p * pomdp_phi_single + (1-p) * mdp_phi
    # state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)
    # lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item(), 'po_type': 'unique'})
    #
    # pomdp.phi = p * pomdp_phi_uniform + (1-p) * mdp_phi
    # state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)
    # lds.append({'p': p, 'ld': discrep_loss(pi_phi, pomdp, alpha=0)[0].item(), 'po_type': 'junction'})

data = pd.DataFrame(lds)

fig = plt.figure(figsize=(12, 3))

key = 'ld'

# Create the left plot (occupying the left half of the figure)
ax = plt.subplot2grid((1, 4), (0, 1), colspan=2, fig=fig)
sns.lineplot(data=data, x='p', y=key, hue='gamma', ax=ax)
ax.set_ylabel(r"")
ax.set_xlabel("")
ax.set_title(f'Lambda Discrepancy ($\Lambda$)')
# ax.semilogy()

# Create the observation model plots
ax = plt.subplot2grid((1, 4), (0, 0), fig=fig)
ax.imshow(mdp_phi, vmin=0, vmax=1)
ax.set_title('Perfect observations')

ax = plt.subplot2grid((1, 4), (0, 3), rowspan=1, colspan=1, fig=fig)
ax.imshow(pomdp_phi_single, vmin=0, vmax=1)
ax.set_title('Aliased observations')


# ax = plt.subplot2grid((3, 5), (1, 0), rowspan=1, colspan=2, fig=fig)
# for gamma in gammas:
#     ld = data.query(f'gamma=={gamma}')[key].to_numpy()
#     ax.plot(ps[1:], (ld[1:] - ld[:-1])/(ps[1]-ps[0]), label=po_type)
# ax.hlines(0, ps[0], ps[-1], 'k', ls='--')
# # ax.set_ylim([-0.001, 0.001])
# ax.set_ylabel(r"$\nabla_p\ \Lambda$")
# ax.set_xlabel(r'Partial Observability ($p$)')
# ax.set_yscale('symlog', linthresh=0.0001)

plt.tight_layout()
plt.savefig('po_sweep_chain_1.png')

plt.show()
