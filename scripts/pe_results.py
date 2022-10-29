import numpy as np
from jax.nn import softmax
from pathlib import Path
from collections import namedtuple
np.set_printoptions(precision=4)

from grl.utils import load_info
from definitions import ROOT_DIR


# %%
results_dir = Path(ROOT_DIR, 'results', 'tmaze_two_thirds_up_mem_grad')

split_by = ['spec', 'algo']
Args = namedtuple('args', split_by)

# %%

all_results = {}

for results_path in results_dir.iterdir():
    if results_path.suffix != '.npy':
        continue
    info = load_info(results_path)
    grad_info = info['grad_info']
    args = info['args']
    single_res = {
        'final_params': grad_info['final_params']
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
RIGHT_ACTION = 2
UP_START_OBS = 0
DOWN_START_OBS = 1
mem_func = jnp.softmax(single_res['final_params'], axis=-1)
right_mem_func = mem_func[RIGHT_ACTION].shape
right_up_start = right_mem_func[UP_START_OBS]
right_down_start = right_mem_func[DOWN_START_OBS]
