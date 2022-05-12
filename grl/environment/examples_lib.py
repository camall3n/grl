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

    phi = np.array([[
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]])

    Pi_phi = [
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 0, 0]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)


def tiger_terminal():
    """
    From: "Acting Optimally in Partially Observable Stochastic Domains"
    https://aaai.org/Papers/AAAI/1994/AAAI94-157.pdf

    Added a terminal state.
    """
    T_listen = np.array([
        [1., 0, 0], # tiger_left state
        [0, 1, 0], # tiger_right state
        [0, 0, 0]
    ])
    T_open = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    T = np.array([T_listen, T_open, T_open]) # listen, open_left, open_right

    R_listen = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ])
    R_open_left = np.array([
        [0, 0, -100], 
        [0, 0, 10],
        [0, 0, 0]
    ])
    R_open_right = np.array([
        [0, 0, 10],
        [0, 0, -100],
        [0, 0, 0]
    ])
    R = np.array([R_listen, R_open_left, R_open_right])

    p0 = np.array([0.5, 0.5, 0])

    phi_listen = np.array([
        [0.85, 0.15, 0],
        [0.15, 0.85, 0],
        [0, 0, 1]
    ])
    phi_open_left = np.array([
        [0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0, 0, 1]
    ])
    phi_open_right = np.array([
        [0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0, 0, 1]
    ])
    phi = np.array([phi_listen, phi_open_left, phi_open_right])

    Pi_phi = [
        np.array([1, 1, 0]),
        np.array([1, 2, 0]),
        np.array([2, 1, 0]),
        np.array([2, 2, 0]),
    ]

    return to_dict(T, R, 0.75, p0, phi, Pi_phi)

def to_dict(T, R, gamma, p0, phi, Pi_phi):
    return {
        'T': T,
        'R': R,
        'gamma': gamma,
        'p0': p0,
        'phi': phi,
        'Pi_phi': Pi_phi,
    }
