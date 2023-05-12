import numpy as np
from pathlib import Path
import inspect

from . import examples_lib
from .memory_lib import get_memory
from .pomdp_file import POMDPFile
from grl.utils.math import normalize
from definitions import ROOT_DIR

def load_spec(name, memory_id: str = None, n_mem_states: int = 2, mem_leakiness: float = 0.1, **kwargs):
    """
    Loads a pre-defined POMDP
    :param name:            The name of the function or .POMDP file defining the POMDP.
    :param memory_id:       ID of memory function to use.
    :param n_mem_states:    Number of memory states allowed.
    :param mem_leakiness:   for memory_id="f" - how leaky do is out leaky identity function.

    The following **kwargs are specified for the following specs:
    tmaze_hyperparams:
        :param corridor_length:     Length of the maze corridor.
        :param discount:            Discount factor gamma to use.
        :param junction_up_pi:      If we specify a policy for the tmaze spec, what is the probability
                                    of traversing UP at the tmaze junction?
    """

    # Try to load from examples_lib first
    # then from pomdp_files
    spec = None
    try:
        spec_fn = getattr(examples_lib, name)
        arg_names = inspect.signature(spec_fn).parameters
        kwargs = {k: v for k, v in kwargs.items() if v is not None and k in arg_names}
        spec = spec_fn(**kwargs)

    except AttributeError as _:
        pass

    if spec is None:
        try:
            file_path = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files', f'{name}.POMDP')
            spec = POMDPFile(file_path).get_spec()
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
                                        n_obs=spec['phi'].shape[-1],
                                        n_actions=spec['T'].shape[0],
                                        n_mem_states=n_mem_states,
                                        leakiness=mem_leakiness)

    # Make sure probs sum to 1
    # e.g. if they are [0.333, 0.333, 0.333], normalizing will do so
    spec['T'] = normalize(spec['T']) # terminal states had all zeros -> nan
    spec['p0'] = normalize(spec['p0'])
    spec['phi'] = normalize(spec['phi'])

    return spec
