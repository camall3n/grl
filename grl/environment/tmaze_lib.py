import numpy as np
from .examples_lib import to_dict

def tmaze(n: int, discount: float = 0.9):
    """
    Return T, R, gamma, p0 and phi for tmaze, for a given corridor length n
    """
    n_states = 2 * n # corridors
    n_states += 2 # start states
    n_states += 2 # T junctions
    n_states += 1 # terminal state

    T_up = np.eye(n_states, n_states)
    T_down = T_up.copy()
    T_up[[-1, -2, -3], [-1, -2, -3]] = 0
    T_down[[-1, -2, -3], [-1, -2, -3]] = 0

    # If we go up or down at the junctions, we terminate
    T_up[[-1 - 1, -2 - 1], [-1, -1]] = 1
    T_down[[-1 - 1, -2 - 1], [-1, -1]] = 1

    T_left = np.zeros((n_states, n_states))
    T_right = T_left.copy()

    # At the leftmost and rightmost states we transition to ourselves
    T_left[[0, 1], [0, 1]] = 1
    T_right[[-1 - 1, -2 - 1], [-1 - 1, -2 - 1]] = 1

    # transition to -2 (left) or +2 (right) index
    all_nonterminal_idxes = np.arange(n_states - 1)
    T_left[all_nonterminal_idxes[2:], all_nonterminal_idxes[:-2]] = 1
    T_right[all_nonterminal_idxes[:-2], all_nonterminal_idxes[2:]] = 1

    T = np.array([T_up, T_right, T_down, T_left])

    # Specify last state as terminal
    T[:, -1, -1] = 1

    R_left = np.zeros((n_states, n_states))
    R_right = R_left.copy()

    R_up = R_left.copy()
    R_down = R_up.copy()

    # If reward is north
    R_up[-2 - 1, -1] = 4
    R_down[-2 - 1, -1] = -0.1

    # If reward is south
    R_up[-1 - 1, -1] = -0.1
    R_down[-1 - 1, -1] = 4

    R = np.array([R_up, R_right, R_down, R_left])

    # Initialize uniformly at random between north start and south start
    p0 = np.zeros(n_states)
    p0[:2] = 0.5

    # Observation function. We have 4 possible obs - start_up, start_down, corridor and junction.
    # terminal obs doesn't matter.
    phi = np.zeros((n_states, 4))

    phi[0, 0] = 1
    phi[1, 1] = 1

    phi[2:-3, 2] = 1

    phi[-3:, 3] = 1

    return T, R, discount, p0, phi

def tmaze_up(n: int, discount: float = 0.9):
    # n_obs x n_actions
    Pi_phi = [np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])]
    return to_dict(*tmaze(n, discount=discount), Pi_phi)

def tmaze_two_thirds_up(n: int, discount: float = 0.9):
    # n_obs x n_actions
    Pi_phi = [np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [2 / 3, 0, 1 / 3, 0]])]
    return to_dict(*tmaze(n, discount=discount), Pi_phi)
