import numpy as np

from . import examples_lib
from .memory_lib import get_memory
from .pomdp_file import POMDPFile

def load_spec(name, memory_id: int = None):
    """
    Loads a pre-defined POMDP
    :param name:      the name of the function or .POMDP file defining the POMDP
    :param memory_id: id of memory function to use
    """

    # Try to load from examples_lib first
    # then from pomdp_files
    spec = None
    try:
        spec = getattr(examples_lib, name)()

    except AttributeError as _:
        pass

    if spec is None:
        try:
            spec = POMDPFile(f'grl/environment/pomdp_files/{name}.POMDP').get_spec()
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

    spec['Pi_phi'] = np.array(spec['Pi_phi']).astype('float')
    if not np.all(len(spec['T']) == np.array([len(spec['R']), len(spec['Pi_phi'][0][0])])):
        raise ValueError("T, R, and Pi_phi must contain the same number of actions")

    if memory_id is not None:
        # TODO: generalize n_mem_states
        spec['mem_params'] = get_memory(memory_id,
                                        spec['phi'].shape[-1],
                                        spec['Pi_phi'].shape[-1],
                                        n_mem_states=2)

    # Make sure probs sum to 1
    # e.g. if they are [0.333, 0.333, 0.333], normalizing will do so
    with np.errstate(invalid='ignore'):
        spec['T'] /= spec['T'].sum(2)[:, :, None]
    spec['T'] = np.nan_to_num(spec['T']) # terminal states had all zeros -> nan
    spec['p0'] /= spec['p0'].sum()

    return spec
