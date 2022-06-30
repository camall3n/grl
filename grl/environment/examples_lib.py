import numpy as np
"""
Library of POMDP specifications. Each function returns a dict of the form:
    {
        T:      transition tensor (terminal states have all outgoing 0s),
        R:      reward tensor,
        gamma:  discount factor,
        p0:     starting state probabilities,
        phi:    observation matrix (currently the same for all actions),
        Pi_phi: policies to evaluate
    }

Functions named 'example_*' come from examples in the GRL workbook.
"""

def example_3():
    # b, r1, r2, t
    T_up = np.array([
        [0., 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    T_down = np.array([
        [0., 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    T = np.array([T_up, T_down])

    R_up = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    R_down = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    p = 0.75
    q = 0.75
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [q, 1 - q],
            [0, 0],
        ]),
    ]

    return to_dict(T, R, 1.0, p0, phi, Pi_phi)

def example_7():
    T = np.array([
        # r, b, r, t
        [0., 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    T = np.array([T, T])

    R = np.array([
        [
            [0., 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
            [1, 0],
        ]),
        np.array([
            [0, 1],
            [0, 1],
            [0, 1],
        ]),
        np.array([
            [4 / 7, 3 / 7], # known location of no discrepancy
            [1, 0],
            [1, 0],
        ])
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_11():
    T = np.array([[
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.5, 0, 0.5],
        [0, 0, 0, 0],
    ]])

    R = np.array([[
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
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
        np.array([
            [1],
            [1],
            [1],
        ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_13():
    T = np.array([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 0]]])

    R = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])

    Pi_phi = [
        np.array([[1], [1], [1]]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_14():
    # b1, b2, r, t
    T_up = np.array([[0., 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    T_down = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    T = np.array([T_up, T_down])

    R_up = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    R_down = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 0.75
    p0[1] = 0.25

    phi = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    p = .5
    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
            [1, 0]
        ]),
        np.array([[0, 1], [0, 1], [0, 1]]),
        np.array([[p, 1 - p], [p, 1 - p], [0, 0]]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_16():
    # b0, b1
    T = np.array([
        [0., 1],
        [1, 0],
    ])
    T = np.array([T, T])

    R_up = np.array([
        [0., 1],
        [0, 0],
    ])
    R_down = np.array([
        [0, 0],
        [1, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1],
        [1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
        ]),
        np.array([
            [0, 1],
        ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_16_terminal():
    gamma_top = 0.5
    # b0, b1, t
    T = np.array([
        [0, gamma_top, 1 - gamma_top],
        [gamma_top, 0, 1 - gamma_top],
        [0, 0, 0],
    ])
    T = np.array([T, T])

    R_up = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    R_down = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
        ]),
        np.array([
            [0, 1],
            [0, 1],
        ]),
    ]

    return to_dict(T, R, 1, p0, phi, Pi_phi)

def example_18():
    T_up = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0], # b
        [0, 0, 0, 1, 0, 0, 0, 0], # r 
        [0, 0, 0, 0, 1, 0, 0, 0], # i
        [0, 0, 0, 0, 0, 1, 0, 0], # y1
        [0, 0, 0, 0, 0, 0, 1, 0], # y2
        [0, 0, 0, 0, 0, 0, 0, 1], # c
        [0, 0, 0, 0, 0, 0, 0, 1], # u
        [0, 0, 0, 0, 0, 0, 0, 0.], # term
    ])
    T_down = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0.],
    ])
    T = np.array([T_up, T_down])

    R = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0.],
    ])
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])

    p = .75
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p]
        ])
    ]

    return to_dict(T, R, 1, p0, phi, Pi_phi)

def example_19():
    T = np.array([[
        [0, 0.5, 0.5, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]])

    R = np.array([[
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
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
        np.array([
            [1],
            [1],
            [1],
        ]),
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
