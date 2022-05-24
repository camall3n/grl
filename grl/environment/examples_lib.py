import numpy as np

"""
Library of POMDP specifications. Each function returns a dict of the form:
    {
        T:      transition tensor,
        R:      reward tensor,
        gamma:  discount factor,
        p0:     starting state probabilities,
        phi:    observation matrix,
        Pi_phi: policies
    }

Functions named 'example_*' come from examples in the GRL workbook.
"""

def example_3():
    # b, r1, r2, t, t, t, t
    p = 0.75
    q = 0.75
    T = np.array([[
        [0, p, 1-p, 0, 0, 0, 0],
        [0, 0, 0, q, 1-q, 0, 0],
        [0, 0, 0, 0, 0, q, 1-q],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]])

    R = np.array([[
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 5, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]])

    Pi_phi = [
        np.array([0, 0, 0, 0, 0, 0, 0]),
    ]
    
    return to_dict(T, R, 1.0, p0, phi, Pi_phi)

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

    phi = np.array([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]])

    Pi_phi = [
        np.array([0, 0, 0, 0]),
    ]
    
    return to_dict(T, R, 0.5, p0, phi, Pi_phi)


def example_13():
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

    phi = np.array([[
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]])

    Pi_phi = [
        np.array([0, 0, 0, 0]),
    ]
    
    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

    
def example_14():
    T_up = np.array([
        [0., 0, 0, 1],
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
    phi = np.array([phi, phi])

    Pi_phi = [
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 0, 0]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)


def to_dict(T, R, gamma, p0, phi, Pi_phi):
    return {
        'T': T,
        'R': R,
        'gamma': gamma,
        'p0': p0,
        'phi': phi,
        'Pi_phi': Pi_phi,
    }
