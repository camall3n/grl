import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from jax.nn import softmax
from pathlib import Path
from collections import namedtuple
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from grl.utils import load_info
from definitions import ROOT_DIR


# %%
results_dir = Path(ROOT_DIR, 'results', 'tmaze_sweep_junction_pi')

split_by = ['spec', 'algo', 'tmaze_corridor_length', 'tmaze_junction_up_pi']
Args = namedtuple('args', split_by)

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
    final_mc_vals, final_td_vals = grad_info['final_vals']['mc'], grad_info['final_vals']['td']
    final_v_discrep = (final_mc_vals['v'] - final_td_vals['v']) ** 2
    final_q_discrep = (final_mc_vals['q'] - final_td_vals['q']) ** 2

    single_res = {
        'final_params': grad_info['final_params'],
        'diff_start_bits_set': diff_start_bits_set,
        'is_flipped': is_flipped,
        'is_set': is_set,
        'is_optimal': diff_start_bits_set and (is_flipped or is_set),
        'tmaze_junction_up_pi': args['tmaze_junction_up_pi'],
        'final_v_discrep': final_v_discrep,
        'final_q_discrep': final_q_discrep,
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
        if k != 'args':
            all_results[hparams][k] = np.stack(v)
# %%

def get_bool_ranges(bools: np.ndarray, x: np.ndarray):
    true_ranges = []
    false_ranges = []

    current_start = x[0]
    current_bool = bools[0]
    for b, potential_end in zip(is_optimals[1:], x[1:]):
        if b != current_bool:
            r = (current_start, potential_end)
            if current_bool:
                true_ranges.append(r)
            else:
                false_ranges.append(r)
            current_start = potential_end
            current_bool = b
    else:
        r = (current_start, potential_end)
        if current_bool:
            true_ranges.append(r)
        else:
            false_ranges.append(r)

    return true_ranges, false_ranges

# %%
pi_map = {}
all_sorted_pis = []
for i, pi in enumerate(sorted(hparam.tmaze_junction_up_pi for hparam in all_results.keys())):
    pi_map[pi] = i
    all_sorted_pis.append(pi)

num_pis = len(all_sorted_pis)
all_to_plot = {
    'discrep_means': np.zeros(num_pis),
    'discrep_std_errs': np.zeros(num_pis),
    'is_optimal_means': np.zeros(num_pis),
    'starting_up_discrep_means': np.zeros((num_pis, 2)),
    'starting_down_discrep_means': np.zeros((num_pis, 2)),
    'starting_up_discrep_std_errs': np.zeros((num_pis, 2)),
    'starting_down_discrep_std_errs': np.zeros((num_pis, 2))
}
for hparam, res in all_results.items():
    all_to_plot['discrep_means'][pi_map[hparam.tmaze_junction_up_pi]] = res['final_v_discrep'].mean(axis=-1).mean(axis=0)
    all_to_plot['discrep_std_errs'][pi_map[hparam.tmaze_junction_up_pi]] = res['final_v_discrep'].mean(axis=-1).std(axis=0) / np.sqrt(res['final_v_discrep'].shape[0])
    all_to_plot['starting_up_discrep_means'][pi_map[hparam.tmaze_junction_up_pi]] = res['final_v_discrep'][:, :2].mean(axis=0)
    all_to_plot['starting_up_discrep_std_errs'][pi_map[hparam.tmaze_junction_up_pi]] = res['final_v_discrep'][:, :2].std(axis=0)
    all_to_plot['is_optimal_means'][pi_map[hparam.tmaze_junction_up_pi]] = res['is_optimal'].mean(axis=0)
res['final_v_discrep'].shape

# %%
# Plots for avg across all values.
fig, ax = plt.figure(), plt.axes()

y_high = 0.065
y_low = 0
x_high = 1
x_low = 0

mean_discreps = all_to_plot['discrep_means']
std_err_discreps = all_to_plot['discrep_std_errs']
mean_is_optimal = all_to_plot['is_optimal_means']
x = all_sorted_pis
ax.imshow(mean_is_optimal[None, :], vmin=0, vmax=1, extent=(x_low, x_high, y_low, y_high), cmap='RdYlGn', aspect='auto', alpha=0.5)
ax.plot(x, mean_discreps, color='blue')
ax.fill_between(x, mean_discreps - std_err_discreps, mean_discreps + std_err_discreps, color='blue', alpha=0.2)

# true_ranges, false_ranges = get_bool_ranges(is_optimals, x)
# for true_begin, true_end in true_ranges:
#     ax.axvspan(true_begin, true_end, alpha=0.2, color='green')
#
# for false_begin, false_end in false_ranges:
#     ax.axvspan(false_begin, false_end, alpha=0.2, color='red')


ax.set_xlabel('π(up | junction)')
ax.set_ylabel('Avg. λ-discreps across obs+')
ax.set_ylim([y_low, y_high])
ax.set_xlim([x_low, x_high])
plt.tight_layout()
# %%
# TODO: plot initial mem lambda discreps
