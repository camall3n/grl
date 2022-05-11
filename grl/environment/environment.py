import logging

import numpy as np

from . import examples_lib
from .pomdp_file import POMDPFile

def load(name):
    """
    Loads a pre-defined POMDP
    :param name: the name of the function defining the POMDP
    """

    # try to load from examples_lib first
    # then from pomdp_files
    try:
        spec = getattr(examples_lib, name)()
    except AttributeError as _:
        try:
            spec = POMDPFile(name).get_spec()
        except FileNotFoundError as _:
            raise NotImplementedError(f'{name} not found in examples_lib.py nor pomdp_files/') from None

    if len(spec) != 6:
        raise ValueError("Expecting POMDP specification of the form: (T, R, gamma, p0, phi, Pi_phi)")
    if len(spec[0].shape) != 3:
        raise ValueError("T tensor must be 3d")
    if len(spec[1].shape) != 3:
        raise ValueError("R tensor must be 3d")

    # Use a random policy if not specified
    if spec[5] is None:
        spec = list(spec)
        spec[5] = [np.random.choice(len(spec[0]), len(spec[0][0]))]

    logging.info(f'T:\n {spec[0]}')
    logging.info(f'R:\n {spec[1]}')
    logging.info(f'gamma: {spec[2]}')
    logging.info(f'p0:\n {spec[3]}')
    logging.info(f'phi:\n {spec[4]}')
    logging.info(f'Pi_phi:\n {spec[5]}')

    return spec
