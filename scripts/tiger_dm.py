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
results_dir = Path(ROOT_DIR, 'results')

split_by = ['spec', 'algo']
Args = namedtuple('args', split_by)

# %%

results_path = results_dir / 'tiger_mi_pi(dm)_miit(1)_s(2023)_Fri Nov 11 16:53:08 2022.npy'
info = load_info(results_path)
args = info['args']
mem_params = info['agent'].mem_params
pi_params = info['agent'].pi_params
epsilon = info['agent'].epsilon
info['logs']['initial_improvement_discrep'], info['logs']['initial_improvement_policy']
# %%

def test_mem_and_pi(mem_params: jnp.ndarray, pi_params: jnp.ndarray, epsilon: float = 0):
    """
    Tests the memory matrix and policy for t-maze.
    our tolerance is set to 1e-1, which seems high, but should be fine for
    stochastic matrices that we have.    info = load_info(results_path)
    grad_info = info['grad_info']
    args = info['args']
    final_mem_params = grad_info['final_params']

    """
    results = {}

    # Memory tests
    RIGHT_ACTION = 2
    UP_START_OBS = 0
    DOWN_START_OBS = 1
    CORRIDOR_OBS = 2
    JUNCTION_OBS = 3

    mem_func = softmax(mem_params, axis=-1)
    right_mem_func = mem_func[RIGHT_ACTION]

    # we index by zero here, since memory starts at 0
    right_up_start = right_mem_func[UP_START_OBS, 0]
    right_down_start = right_mem_func[DOWN_START_OBS, 0]

    # we test whether start bits set to different memory states
    def test_start_bits_set(right_up_start: np.ndarray, right_down_start: np.ndarray):
        return np.isclose(np.abs(right_up_start - right_down_start).sum() / 2, 1, atol=1e-1)

    diff_start_mem_bits_set = test_start_bits_set(right_up_start, right_down_start)
    results['diff_start_mem_bits_set'] = diff_start_mem_bits_set

    # now we test whether the right corridor memory function is all set or reset
    right_corridor = right_mem_func[CORRIDOR_OBS]

    def test_corridor_hold_or_flip(right_corridor: np.ndarray):
        is_flipped = np.allclose(right_corridor, np.eye(2)[:, ::-1], atol=1e-1)
        is_set = np.allclose(right_corridor, np.eye(2), atol=1e-1)
        return is_flipped, is_set

    is_flipped_mem, is_set_mem = test_corridor_hold_or_flip(right_corridor)
    results['is_flipped_mem'], results['is_set_mem'] = is_flipped_mem, is_set_mem

    # Policy tests
    policy = softmax(pi_params, axis=-1)

    # first we make sure we go right in the right spots
    pi_begin_right = np.allclose(policy[:6, 2], 1 - epsilon + (epsilon / pi_params.shape[-1]))

    results['pi_begin_right'] = pi_begin_right

    pi_junction = policy[6:8]
    pi_junction_0_mem, pi_junction_1_mem = pi_junction[:, :2]
    # right_up_start, right_down_start: what are we setting the memory bit to in the beginning?
    # pi_junction_0_mem, pi_junction_1_mem: what is the policy at the junction, given the memory bit?

    # we need to see if pi_junction_0_mem == right_up_start, and pi_junction_1_mem == right_down_start
    # TODO: think about whether or not if we flip this it might be ok as well
    up_mem_matches_junc_pi = np.allclose(right_up_start[pi_junction_0_mem.argmax()], 1, atol=1e-1)
    down_mem_matches_junc_pi = np.allclose(right_down_start[pi_junction_1_mem.argmax()],
                                           1,
                                           atol=1e-1)

    results['up_mem_matches_junc_pi'] = up_mem_matches_junc_pi
    results['down_mem_matches_junc_pi'] = down_mem_matches_junc_pi

    results['is_optimal'] = up_mem_matches_junc_pi and down_mem_matches_junc_pi and pi_begin_right
    return results

# %%
all_results = {}

for results_path in results_dir.iterdir():
    if results_path.suffix != '.npy':
        continue
    results_path = list(results_dir.iterdir())[0]
    info = load_info(results_path)
    args = info['args']
    mem_params = info['agent'].mem_params
    pi_params = info['agent'].pi_params
    epsilon = info['agent'].epsilon

    initial_q_discrep, initial_v_discrep = info['logs']['initial_mem_discrep']['q'], info['logs'][
        'initial_mem_discrep']['v']
    final_q_discrep, final_v_discrep = info['logs']['final_mem_discrep']['q'], info['logs'][
        'final_mem_discrep']['v']

    results = test_mem_and_pi(mem_params, pi_params, epsilon)

    single_res = {
        'initial_v_discrep': initial_v_discrep,
        'initial_q_discrep': initial_q_discrep,
        'final_v_discrep': final_v_discrep,
        'final_q_discrep': final_q_discrep,
    }
    for k, v in results.items():
        single_res[k] = v

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
all_results[hparams]['is_optimal']

# %%
pi_map = {}
all_sorted_pis = []
for i, pi in enumerate(sorted(hparam.tmaze_junction_up_pi for hparam in all_results.keys())):
    pi_map[pi] = i
    all_sorted_pis.append(pi)

num_pis = len(all_sorted_pis)
all_to_plot = {
    'final_discrep_means': np.zeros(num_pis),
    'final_discrep_std_errs': np.zeros(num_pis),
    'initial_discrep_means': np.zeros(num_pis),
    'initial_discrep_std_errs': np.zeros(num_pis),
    'is_optimal_means': np.zeros(num_pis),
    'starting_up_discrep_means': np.zeros((num_pis, 2)),
    'starting_down_discrep_means': np.zeros((num_pis, 2)),
    'starting_up_discrep_std_errs': np.zeros((num_pis, 2)),
    'starting_down_discrep_std_errs': np.zeros((num_pis, 2))
}
for hparam, res in all_results.items():
    all_to_plot['final_discrep_means'][pi_map[
        hparam.tmaze_junction_up_pi]] = res['final_v_discrep'].mean(axis=-1).mean(axis=0)
    all_to_plot['final_discrep_std_errs'][pi_map[
        hparam.tmaze_junction_up_pi]] = res['final_v_discrep'].mean(axis=-1).std(axis=0) / np.sqrt(
            res['final_v_discrep'].shape[0])
    all_to_plot['initial_discrep_means'][pi_map[
        hparam.tmaze_junction_up_pi]] = res['initial_v_discrep'].mean(axis=-1).mean(axis=0)
    all_to_plot['initial_discrep_std_errs'][pi_map[
        hparam.tmaze_junction_up_pi]] = res['initial_v_discrep'].mean(axis=-1).std(
            axis=0) / np.sqrt(res['initial_v_discrep'].shape[0])
    all_to_plot['starting_up_discrep_means'][pi_map[
        hparam.tmaze_junction_up_pi]] = res['final_v_discrep'][:, :2].mean(axis=0)
    all_to_plot['starting_up_discrep_std_errs'][pi_map[
        hparam.tmaze_junction_up_pi]] = res['final_v_discrep'][:, :2].std(axis=0)
    all_to_plot['is_optimal_means'][pi_map[hparam.tmaze_junction_up_pi]] = res['is_optimal'].mean(
        axis=0)
res['final_v_discrep'].shape

# %%
# Plots for avg across all values.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

y_high = 0.25
y_low = 0
x_high = 1
x_low = 0

final_discrep_means = all_to_plot['final_discrep_means']
final_discrep_std_errs = all_to_plot['final_discrep_std_errs']
initial_discrep_means = all_to_plot['initial_discrep_means']
initial_discrep_std_errs = all_to_plot['initial_discrep_std_errs']
mean_is_optimal = all_to_plot['is_optimal_means']
x = all_sorted_pis

axes[0].plot(x, initial_discrep_means, color='blue')
axes[0].fill_between(x,
                     initial_discrep_means - initial_discrep_std_errs,
                     initial_discrep_means + initial_discrep_std_errs,
                     color='blue',
                     alpha=0.2)
axes[0].set_title('Initial λ-discreps')
# axes[0].set_xlabel('π(up | junction)')
axes[0].set_ylabel('Avg. λ-discreps across obs+')
axes[0].set_ylim([y_low, y_high])
axes[0].set_xlim([x_low, x_high])

axes[1].imshow(mean_is_optimal[None, :],
               vmin=0,
               vmax=1,
               extent=(x_low, x_high, y_low, y_high),
               cmap='RdYlGn',
               aspect='auto',
               alpha=0.5)
axes[1].plot(x, final_discrep_means, color='blue')
axes[1].fill_between(x,
                     final_discrep_means - final_discrep_std_errs,
                     final_discrep_means + final_discrep_std_errs,
                     color='blue',
                     alpha=0.2)
axes[1].set_title('Final λ-discreps')

# axes[1].set_xlabel('π(up | junction)')
# axes[1].set_ylabel('Avg. λ-discreps across obs+')
axes[1].set_ylim([y_low, y_high])
axes[1].set_xlim([x_low, x_high])
fig.text(0.52, 0.0, 'π(up | junction)', ha='center', va='center')

plt.tight_layout()
# %%
