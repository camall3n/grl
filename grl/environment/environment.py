import numpy as np
from pathlib import Path

from . import examples_lib
from .memory_lib import get_memory
from .pomdp_file import POMDPFile
from grl.utils import normalize
from definitions import ROOT_DIR

def load_spec(name, *args, memory_id: int = None, n_mem_states: int = 2, **kwargs):
    """
    Loads a pre-defined POMDP
    :param name:      the name of the function or .POMDP file defining the POMDP
    :param memory_id: id of memory function to use
    """

    # Try to load from examples_lib first
    # then from pomdp_files
    spec = None
    try:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        spec = getattr(examples_lib, name)(*args, **kwargs)

    except AttributeError as _:
        pass

    if spec is None:
        try:
            file_path = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files', f'{name}.POMDP')
            spec = POMDPFile(file_path).get_spec(*args, **kwargs)
        except FileNotFoundError as _:
            raise NotImplementedError(
                f'{name} not found in examples_lib.py nor pomdp_files/') from None

    # Check sizes and types
    if len(spec.keys()) < 6:
        raise ValueError("POMDP specification must contain at least: T, R, gamma, p0, phi, Pi_phi")
    if len(spec['T'].shape) != 3:
        raise ValueError("T tensor must be 3d")
    if len(spec['R'].shape) != 3:
        raise ValueError("R tensor must be 3d")

    if spec['Pi_phi'] is not None:
        spec['Pi_phi'] = np.array(spec['Pi_phi']).astype('float')
        spec['Pi_phi'] = normalize(spec['Pi_phi'])
        if not np.all(len(spec['T']) == np.array([len(spec['R']), len(spec['Pi_phi'][0][0])])):
            raise ValueError("T, R, and Pi_phi must contain the same number of actions")

    if memory_id is not None:
        spec['mem_params'] = get_memory(memory_id,
                                        spec['phi'].shape[-1],
                                        spec['T'].shape[0],
                                        n_mem_states=n_mem_states)

    # Make sure probs sum to 1
    # e.g. if they are [0.333, 0.333, 0.333], normalizing will do so
    spec['T'] = normalize(spec['T']) # terminal states had all zeros -> nan
    spec['p0'] = normalize(spec['p0'])
    spec['phi'] = normalize(spec['phi'])

    return spec
