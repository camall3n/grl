# %% codecell
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import numpy as np

from jax.config import config
from jax.nn import softmax
from pathlib import Path
config.update('jax_platform_name', 'cpu')

from grl.utils import load_info
from grl.utils.mdp import get_perf
from definitions import ROOT_DIR
# %% codecell
results_dir = Path(ROOT_DIR, 'results', 'tmaze_sweep_alpha')

args_to_extract = ['spec', 'algo', 'n_mem_states', 'alpha', 'seed']
group_by_args = [arg for arg in args_to_extract if arg != 'seed']
# %% codecell
results_path = list(results_dir.iterdir())[40]


info = load_info(results_path)

# %% codecell

def test_mem_matrix(mem_params: jnp.ndarray):
    """
    Tests the memory matrix for t-maze.
    our tolerance is set to 1e-1, which seems high, but should be fine for
    stochastic matrices that we have.
    """
    RIGHT_ACTION = 2
    UP_START_OBS = 0
    DOWN_START_OBS = 1
    CORRIDOR_OBS = 2

    mem_func = softmax(mem_params, axis=-1)
    right_mem_func = mem_func[RIGHT_ACTION]

    # we index by zero here, since memory starts at 0
    right_up_start = right_mem_func[UP_START_OBS, 0]
    right_down_start = right_mem_func[DOWN_START_OBS, 0]

    # we test whether start bits set to different memory states
    def test_start_bits_set(right_up_start: np.ndarray, right_down_start: np.ndarray):
        return (np.abs(right_up_start - right_down_start).sum() / 2 - 1)**2

    diff_start_bits_set = test_start_bits_set(right_up_start, right_down_start)

    # now we test whether the right corridor memory function is all set or reset
    right_corridor = right_mem_func[CORRIDOR_OBS]

    def test_corridor_hold_or_toggle(right_corridor: np.ndarray):
        is_toggle = ((right_corridor - np.eye(2)[:, ::-1])**2).mean()
        is_hold = ((right_corridor - np.eye(2))**2).mean()
        return is_toggle, is_hold

    is_toggle, is_hold = test_corridor_hold_or_toggle(right_corridor)
    return diff_start_bits_set.item(), is_toggle.item(), is_hold.item()

list_for_df = []

for results_path in list(results_dir.iterdir()):
    if results_path.is_dir() or results_path.suffix != '.npy':
        continue

    info = load_info(results_path)
    agent_path = Path(results_path.parent, 'agents', f"{results_path.stem}.pkl.npy")
    agent = load_info(agent_path)

    args = info['args']

    # agent = info['agent']
    init_policy_info = info['logs']['initial_policy_stats']
    final_mem_info = info['logs']['greedy_final_mem_stats']

    diff_start_bits_set, is_toggle, is_hold = test_mem_matrix(agent.mem_params)

    single_row = {k: args[k] for k in args_to_extract}

    single_row.update({
        'init_policy_perf': init_policy_info['discrep'].item(),
        'final_mem_perf': final_mem_info['discrep'].item(),
        'diff_start_bits_set': diff_start_bits_set,
        'is_toggle': is_toggle,
        'is_hold': is_hold,
        'is_optimal': 0.5 * diff_start_bits_set + 0.5 * min(is_toggle, is_hold),
        # 'final_mem': np.array(agent.memory),
        # 'final_policy': np.array(agent.policy)
    })
    list_for_df.append(single_row)

df = pd.DataFrame(list_for_df)
# %% codecell
df.groupby(group_by_args).mean()
# %% codecell
