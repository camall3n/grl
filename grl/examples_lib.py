import sys 

import numpy as np

def load(name):
    """
    Loads a pre-defined POMDP
    :param name: the name of the function defining the POMDP
    """

    spec = getattr(sys.modules[__name__], name)()
    if len(spec) != 6:
        raise ValueError("Expecting POMDP specification of the form: (T, R, gamma, p0, phi, Pi_phi)")
    if len(spec[0].shape) != 3:
        raise ValueError("T tensor must be 3d")
    if len(spec[1].shape) != 3:
        raise ValueError("R tensor must be 3d")

    return spec

def example_11():
    T = np.array([[
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.5, 0, 0.5],
        [0, 0, 0, 0]
    ]])

    R = np.array([[
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    Pi_phi = [
        np.array([0, 0, 0, 0]),
    ]
    
    return T, R, 0.5, p0, phi, Pi_phi

def example_13():
    T_up = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    T_down = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    T = np.array([T_up, T_down])

    R = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 0.75
    p0[1] = 0.25

    phi = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    Pi_phi = [
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 0, 0]),
    ]

    return T, R, 0.5, p0, phi, Pi_phi
