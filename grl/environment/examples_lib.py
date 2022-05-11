import numpy as np

"""
Library of POMDP specifications. Each function returns a tuple of the form:
    (
        T:      transition tensor,
        R:      reward tensor,
        gamma:  discount factor,
        p0:     starting state probabilities,
        phi:    observation matrix,
        Pi_phi: policies
    )

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

    
def example_14():
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


def tiger():
    """
    From: "Acting Optimally in Partially Observable Stochastic Domains"
    https://aaai.org/Papers/AAAI/1994/AAAI94-157.pdf
    """
    T_listen = np.array([
        [1, 0], # tiger_left state
        [0, 1]  # tiger_right state
    ])
    T_open_left = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    T_open_right = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    T = np.array([T_listen, T_open_left, T_open_right])

    R_listen = np.array([
        [-1, -1],
        [-1, -1]
    ])
    # For rewards after "open" actions, only the start state and action matter
    # (the game resets after picking a door so the next state is random)
    R_open_left = np.array([
        [-100, -100], 
        [10, 10]
    ])
    R_open_right = np.array([
        [10, 10],
        [-100, -100]
    ])
    R = np.array([R_listen, R_open_left, R_open_right])

    p0 = 0.5 * np.ones(len(T[0]))

    phi = np.array([
        [0.85, 0.15],
        [0.15, 0.85],
    ])

    Pi_phi = [
        np.array([1, 1]),
        np.array([1, 2]),
        np.array([2, 1]),
        np.array([2, 2]),
    ]

    return T, R, 0.75, p0, phi, Pi_phi
