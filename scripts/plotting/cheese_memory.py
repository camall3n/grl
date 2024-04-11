# %% codecell
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from jax.nn import softmax
from jax.config import config
from pathlib import Path
from collections import namedtuple

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

from grl.utils import load_info
from grl.utils.mdp import get_perf
from definitions import ROOT_DIR
np.set_printoptions(precision=3)
# %% codecell

results_dir = Path(ROOT_DIR, 'results')

# %% codecell
all_results = {}

# for results_path in results_dir.iterdir():
#     if 'tmaze' not in results_path.stem:
#         continue
#     info = load_info(results_path)
# results_paths = [res_path for res_path in results_dir.iterdir() if 'cheese' in res_path.stem]
# results_path = results_paths[0]
results_path = results_dir / 'cheese.95_mi_pi(pi)_miit(1)_s(2020)_Thu Feb  2 20:50:49 2023.npy'
# %% codecell
agent_path = results_path.parent / 'agents' / f"{results_path.stem}.pkl.npy"
# %% codecell
agent = load_info(agent_path)
mem = agent.memory
action_map = ['NORTH', 'SOUTH', 'EAST', 'WEST']
# info['logs']['initial_improvement_policy'], agent.memory, agent.policy
for a, m in enumerate(mem):
    print(action_map[a])
    print(mem)
    print()
# %% codecell
tol = 0.1

SET = np.array([
    [0, 1],
    [0, 1]
])

RESET = np.array([
    [1, 0],
    [1, 0]
])

HOLD = np.array([
    [1, 0],
    [0, 1]
])

FLIP = np.array([
    [0, 1],
    [1, 0]
])

for a, act_mem in enumerate(agent.memory):
    for obs_idx, obs_act_mem in enumerate(act_mem):
        if obs_idx == act_mem.shape[0] - 1:
            continue

        if np.allclose(obs_act_mem, SET, atol=tol):
            print(f"(obs: {obs_idx}, action: {action_map[a]}) -> SET")
        elif np.allclose(obs_act_mem, RESET, atol=tol):
            print(f"(obs: {obs_idx}, action: {action_map[a]}) -> RESET")
        elif np.allclose(obs_act_mem, HOLD, atol=tol):
            print(f"(obs: {obs_idx}, action: {action_map[a]}) -> HOLD")
        elif np.allclose(obs_act_mem, FLIP, atol=tol):
            print(f"(obs: {obs_idx}, action: {action_map[a]}) -> FLIP")
        else:
            continue

# %% codecell

# %% markdown
# (obs: 1, action: NORTH) -> HOLD
# (obs: 5, action: NORTH) -> SET
# (obs: 0, action: SOUTH) -> SET
# (obs: 1, action: SOUTH) -> HOLD
# (obs: 2, action: SOUTH) -> RESET
# (obs: 3, action: SOUTH) -> SET
# (obs: 0, action: EAST) -> SET
# (obs: 4, action: EAST) -> HOLD
# (obs: 3, action: WEST) -> RESET
# (obs: 4, action: WEST) -> HOLD
# %% codecell
