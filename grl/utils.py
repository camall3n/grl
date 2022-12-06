import numpy as np
import jax.numpy as jnp
from pathlib import Path
from time import time, ctime
from argparse import Namespace

from pprint import pformat
from typing import Sequence, Union, Tuple
from definitions import ROOT_DIR

RTOL = 1e-3

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)

def results_path(args: Namespace):
    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)
    if args.experiment_name is not None:
        results_dir /= args.experiment_name
    results_dir.mkdir(exist_ok=True)
    if args.algo == 'mi':
        results_path = results_dir / f"{args.spec}_{args.algo}_pi({args.policy_optim_alg})_miit({args.mi_iterations})_s({args.seed})_{ctime(time())}.npy"
    elif args.algo == 'pe':
        results_path = results_dir / f"{args.spec}_{args.algo}_method({args.method})_grad({args.use_grad})_s({args.seed})_{ctime(time())}.npy"
    elif args.algo == 'vi':
        results_path = results_dir / f"{args.spec}_{args.algo}_s({args.seed})_{ctime(time())}.npy"
    else:
        raise NotImplementedError
    return results_path

def glorot_init(shape: Sequence[int], scale: float = 0.5) -> jnp.ndarray:
    return np.random.normal(size=shape) * scale

def numpyify_dict(info: Union[dict, jnp.ndarray, np.ndarray, list, tuple]):
    """
    Converts all jax.numpy arrays to numpy arrays in a nested dictionary.
    """
    if isinstance(info, jnp.ndarray):
        return np.array(info)
    elif isinstance(info, dict):
        return {k: numpyify_dict(v) for k, v in info.items()}
    elif isinstance(info, list):
        return [numpyify_dict(i) for i in info]
    elif isinstance(info, tuple):
        return (numpyify_dict(i) for i in info)

    return info

def numpyify_and_save(path: Path, info: dict):
    numpy_dict = numpyify_dict(info)
    np.save(path, numpy_dict)

def load_info(results_path: Path) -> dict:
    return np.load(results_path, allow_pickle=True).item()

def normalize(arr: np.ndarray, axis=-1) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        normalized_arr = arr / np.expand_dims(arr.sum(axis), axis)
    normalized_arr = np.nan_to_num(normalized_arr)
    return normalized_arr

def greedify(pi: np.ndarray) -> np.ndarray:
    """
    pi: |O| x |A| array.
    This function returns an array of the same size, except with the max values of each row
    == 1, where everything else is 0.
    """
    pi_greedy = np.zeros_like(pi)
    pi_greedy[np.arange(pi.shape[0]), pi.argmax(axis=-1)] = 1
    return pi_greedy



