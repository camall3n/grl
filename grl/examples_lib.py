import numpy as np


def example_11():
    T = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.5, 0, 0.5],
        [0, 0, 0, 0]
    ])

    R = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    p0 = np.zeros(T.shape[0])
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
