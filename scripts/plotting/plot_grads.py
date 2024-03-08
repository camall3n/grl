from pathlib import Path
from functools import partial

from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from grl.utils.file_system import load_info

ACTION_IDX = 2
OBS_IDXES = slice(0, 3)
OBS_TITLES = ('UP', 'DOWN', 'HALLWAY')

def plot_gradients_of_logits(results_path, plot_first: int = None):

    results = load_info(results_path)
    mem_op_info = results['logs']['after_mem_op']['update_logs']

    grads = mem_op_info['grad'][0, :, ACTION_IDX, OBS_IDXES]
    grad_min, grad_max = grads.min(), grads.max()

    if plot_first is not None:
        grads = grads[:plot_first]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axslider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax=axslider, label='Index', valmin=0, valmax=grads.shape[0] - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        # Go through
        for i, gradients in enumerate(grads[idx]):
            # Plot the gradients
            # Using a diverging colormap to accommodate both positive and negative values
            axs[i].clear()
            im = axs[i].imshow(gradients, cmap='coolwarm', vmin=grad_min, vmax=grad_max)

            # Adding grid lines
            axs[i].grid(which='major', color='black', linestyle='-', linewidth=2)
            axs[i].set_xticks(np.arange(-.5, 2, 1), minor=True)
            axs[i].set_yticks(np.arange(-.5, 2, 1), minor=True)
            axs[i].grid(which='minor', color='black', linestyle='-', linewidth=2)

            axs[i].set_title(OBS_TITLES[i])

            # Hide the ticks
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        return im

    slider.on_changed(update)
    first_im = update(0)

    # Adding a color bar to indicate the scale
    fig.colorbar(first_im, ax=axs.ravel().tolist(), shrink=0.9)
    fig.suptitle('Gradients of Memory Logits')

    plt.show()

def plot_stochastic_matrices_with_grid(results_path, plot_first: int = None):
    results = load_info(results_path)
    mem_op_info = results['logs']['after_mem_op']['update_logs']
    mems = softmax(mem_op_info['intermediate_mem'][0, :, ACTION_IDX, OBS_IDXES], axis=-1)
    if plot_first is not None:
        mems = mems[:plot_first]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axslider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax=axslider, label='Index', valmin=0, valmax=mems.shape[0] - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        for i, mem in enumerate(mems[idx]):
            # Check if the matrix is stochastic (rows sum to 1)
            if not np.allclose(mem.sum(axis=1), np.ones(mem.shape[0])):
                raise ValueError(f"Matrix {i + 1} is not stochastic.")

            # Plot the matrix with grid lines
            axs[i].imshow(mem, cmap='gray', vmin=0, vmax=1)

    # Adding grid lines
    for i in range(len(axs)):
        axs[i].grid(which='major', color='black', linestyle='-', linewidth=2)
        axs[i].set_xticks(np.arange(-.5, 2, 1), minor=True)
        axs[i].set_yticks(np.arange(-.5, 2, 1), minor=True)
        axs[i].grid(which='minor', color='black', linestyle='-', linewidth=2)

        axs[i].set_title(OBS_TITLES[i])

        # Hide the ticks
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    slider.on_changed(update)
    update(0)

    plt.show()


def visualize_matrix_stack_corrected(matrix_stack):
    # Initial setup
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Define a placeholder update function
    def update_placeholder(val):
        pass

    # Slider
    axslider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax=axslider, label='Index', valmin=0, valmax=len(matrix_stack) - 1, valinit=0, valstep=1)

    # Proper update function, now that slider is defined
    def update(val):
        idx = int(slider.val)
        for i in range(3):
            axs[i].clear()
            axs[i].imshow(matrix_stack[idx, i], cmap='viridis', vmin=0, vmax=1)
            axs[i].set_title(f'Matrix {i + 1}, Index: {idx}')
            axs[i].grid(which='major', color='white', linestyle='-', linewidth=0.5)
            axs[i].set_xticks(np.arange(-.5, 2, 1), minor=True)
            axs[i].set_yticks(np.arange(-.5, 2, 1), minor=True)
            axs[i].grid(which='minor', color='white', linestyle='-', linewidth=0.5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])

    slider.on_changed(update)

    # Display the first set of matrices
    update(0)

    plt.show()

def get_plot_fns(results_path,
                 action_idx: int = 2,
                 obs_idxes: slice = slice(0, 3),
                 obs_titles: tuple = ('UP', 'DOWN', 'HALLWAY')):
    results = load_info(results_path)
    mem_op_info = results['logs']['after_mem_op']['update_logs']

    grads = mem_op_info['grad'][0]
    grad_min, grad_max = grads.min(), grads.max()

    mems = softmax(mem_op_info['intermediate_mem'][0], axis=-1)

    # mem = mems[0][ACTION_IDX][OBS_IDXES]
    # grad = grads[0][ACTION_IDX][OBS_IDXES]

    # plot_stochastic_matrices_with_grid(mem, obs_titles)
    # plot_gradients_of_logits(grad, obs_titles, vmin=grad_min, vmax=grad_max)

    def plot_grads_fn(frame_number):
        plt.clf()
        grad = grads[frame_number][action_idx][obs_idxes]
        plot_gradients_of_logits(grad, obs_titles, vmin=grad_min, vmax=grad_max)

    def plot_mem_fn(frame_number):
        plt.clf()
        mem = mems[frame_number][action_idx][obs_idxes]
        plot_stochastic_matrices_with_grid(mem, obs_titles)

    return plot_grads_fn, plot_mem_fn, \
        {'grads_max_t': grads.shape[0], 'mem_max_t': mems.shape[0]}

if __name__ == "__main__":
    # Lambda discrep
    # results_path = Path('/Users/ruoyutao/Documents/grl/results/tmaze_5_two_thirds_up_batch_run_seed(2020)_time(20240222-190925)_e58205e91751c77f3d7e7588d1db094b.npy')
    results_path = Path('/Users/ruoyutao/Documents/grl/results/tmaze_5_two_thirds_up_batch_run_seed(2026)_time(20240222-191218)_8eff798d802246f1817333465051006d.npy')

    # broken lambda discrep
    # results_path = Path('/Users/ruoyutao/Documents/grl/results/tmaze_5_two_thirds_up_batch_run_seed(2022)_time(20240223-112401)_faa1d82d2611d80b1f129ee4f652a874.npy')

    # TDE
    # results_path = Path('/Users/ruoyutao/Documents/grl/results/tmaze_5_two_thirds_up_batch_run_seed(2026)_time(20240222-193157)_4bbc8e57d9bcb06b407a5676ebf470e7.npy')

    plot_gradients_of_logits(results_path, plot_first=300)
    # plot_stochastic_matrices_with_grid(results_path, plot_first=300)
