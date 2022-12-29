import numpy as np
import jax.numpy as jnp
from jax.nn import softmax
from pathlib import Path
from collections import namedtuple

np.set_printoptions(precision=4)

from grl.utils import load_info
from definitions import ROOT_DIR

# %%
results_dir = Path(ROOT_DIR, 'results')
# results_dir = Path(ROOT_DIR, 'results', 'slippery_tmaze_two_thirds_up_mem_grad')

split_by = ['spec', 'algo']
Args = namedtuple('args', split_by)
results_dir

# %%
results_path = results_dir / 'tmaze_eps_hyperparams_pe_method(a)_grad(m)_s(2022)_Mon Nov  7 14:47:10 2022.npy'
info = load_info(results_path)
info
# %%

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
        return np.isclose(np.abs(right_up_start - right_down_start).sum() / 2, 1, atol=1e-1)

    diff_start_bits_set = test_start_bits_set(right_up_start, right_down_start)

    # now we test whether the right corridor memory function is all set or reset
    right_corridor = right_mem_func[CORRIDOR_OBS]

    def test_corridor_hold_or_flip(right_corridor: np.ndarray):
        is_flipped = np.allclose(right_corridor, np.eye(2)[:, ::-1], atol=1e-1)
        is_set = np.allclose(right_corridor, np.eye(2), atol=1e-1)
        return is_flipped, is_set

    is_flipped, is_set = test_corridor_hold_or_flip(right_corridor)
    return diff_start_bits_set, is_flipped, is_set

# %%
all_results = {}

for results_path in results_dir.iterdir():
    if results_path.suffix != '.npy':
        continue
    info = load_info(results_path)
    grad_info = info['grad_info']
    args = info['args']
    final_mem_params = grad_info['final_params']
    diff_start_bits_set, is_flipped, is_set = test_mem_matrix(final_mem_params)

    single_res = {
        'final_params': grad_info['final_params'],
        'diff_start_bits_set': diff_start_bits_set,
        'is_flipped': is_flipped,
        'is_set': is_set,
        'is_optimal': diff_start_bits_set and (is_flipped or is_set)
    }

    hparams = Args(*tuple(args[s] for s in split_by))

    if hparams not in all_results:
        all_results[hparams] = {}

    for k, v in single_res.items():
        if k not in all_results[hparams]:
            all_results[hparams][k] = []
        all_results[hparams][k].append(v)
    all_results[hparams]['args'] = args

for hparams, res_dict in all_results.items():
    for k, v in res_dict.items():
        all_results[hparams][k] = np.stack(v)
info['grad_info'].keys()

# %%

all_results[list(all_results.keys())[0]]['final_params'].shape

# %%
run_data = all_results[list(all_results.keys())[0]]
run_data['is_optimal'], run_data['diff_start_bits_set'], run_data['is_flipped'], run_data['is_set']

# %%
all_mem_params = all_results[list(all_results.keys())[0]]['final_params']
idx = 2

RIGHT_ACTION = 2
UP_START_OBS = 0
DOWN_START_OBS = 1
CORRIDOR_OBS = 2

mem_func = softmax(all_mem_params[idx], axis=-1)
right_mem_func = mem_func[RIGHT_ACTION]

# we index by zero here, since memory starts at 0
right_up_start = right_mem_func[UP_START_OBS, 0]
right_down_start = right_mem_func[DOWN_START_OBS, 0]

# we test whether start bits set to different memory states
right_up_start - right_down_start
np.isclose(np.abs(right_up_start - right_down_start).sum() / 2, 1, atol=1e-1)

def test_start_bits_set(right_up_start: np.ndarray, right_down_start: np.ndarray):
    return np.isclose(np.abs(right_up_start - right_down_start).sum() / 2, 1, atol=1e-1)

diff_start_bits_set = test_start_bits_set(right_up_start, right_down_start)

# now we test whether the right corridor memory function is all set or reset
right_corridor = right_mem_func[CORRIDOR_OBS]

def test_corridor_hold_or_flip(right_corridor: np.ndarray):
    is_flipped = np.allclose(right_corridor, np.eye(2)[:, ::-1], atol=1e-2)
    is_set = np.allclose(right_corridor, np.eye(2), atol=1e-2)
    return is_flipped, is_set

is_flipped, is_set = test_corridor_hold_or_flip(right_corridor)
