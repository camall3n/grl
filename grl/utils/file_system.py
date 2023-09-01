from argparse import Namespace
import hashlib
import importlib
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import time
from typing import Union

from definitions import ROOT_DIR

def results_path(args: Namespace):
    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)

    args_hash = make_hash_md5(args.__dict__)
    time_str = time.strftime("%Y%m%d-%H%M%S")

    if args.study_name is not None:
        results_dir /= args.study_name
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.spec}_seed({args.seed})_time({time_str})_{args_hash}.npy"
    return results_path

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
        return tuple(numpyify_dict(i) for i in info)

    return info

def numpyify_and_save(path: Path, info: Union[dict, jnp.ndarray, np.ndarray, list, tuple]):
    numpy_dict = numpyify_dict(info)
    np.save(path, numpy_dict)

def load_info(results_path: Path) -> dict:
    return np.load(results_path, allow_pickle=True).item()

def make_hash_md5(o):
    return hashlib.md5(str(o).encode('utf-8')).hexdigest()

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def import_module_to_var(fpath: Path, var_name: str) -> Union[dict, list]:
    spec = importlib.util.spec_from_file_location(var_name, fpath)
    var_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(var_module)
    instantiated_var = getattr(var_module, var_name)
    return instantiated_var
