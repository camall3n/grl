import numpy as np
from grl.utils.math import normalize

START_UP = 0
START_DOWN = 1
JUNCTION_UP = -3
JUNCTION_DOWN = -2
TERMINAL = -1

def tmaze(n: int,
          discount: float = 0.9,
          good_term_reward: float = 4.0,
          bad_term_reward: float = -0.1):
    """
    Return T, R, gamma, p0 and phi for tmaze, for a given corridor length n

                            +---+
          X                 | G |     S: Start
        +---+---+---+   +---+---+     X: Goal Indicator
        | S |   |   |...|   | J |     J: Junction
        +---+---+---+   +---+---+     G: Terminal (Goal)
          0   1   2       n | T |     T: Terminal (Non-Goal)
                            +---+
    """
    n_states = 2 * n # Corridor
    n_states += 2 # Start
    n_states += 2 # Junction
    n_states += 1 # Terminal state (includes both goal/non-goal)

    T_up = np.eye(n_states, n_states)
    T_down = T_up.copy()
    T_up[[TERMINAL, JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, JUNCTION_DOWN, JUNCTION_UP]] = 0
    T_down[[TERMINAL, JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, JUNCTION_DOWN, JUNCTION_UP]] = 0

    # If we go up or down at the junctions, we terminate
    T_up[[JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, TERMINAL]] = 1
    T_down[[JUNCTION_DOWN, JUNCTION_UP], [TERMINAL, TERMINAL]] = 1

    T_left = np.zeros((n_states, n_states))
    T_right = T_left.copy()

    # At the leftmost and rightmost states we transition to ourselves
    T_left[[START_UP, START_DOWN], [START_UP, START_DOWN]] = 1
    T_right[[JUNCTION_DOWN, JUNCTION_UP], [JUNCTION_DOWN, JUNCTION_UP]] = 1

    # transition to -2 (left) or +2 (right) index
    all_nonterminal_idxes = np.arange(n_states - 1)
    T_left[all_nonterminal_idxes[2:], all_nonterminal_idxes[:-2]] = 1
    T_right[all_nonterminal_idxes[:-2], all_nonterminal_idxes[2:]] = 1

    T = np.array([T_up, T_down, T_right, T_left])

    # Specify last state as terminal
    T[:, TERMINAL, TERMINAL] = 1

    R_left = np.zeros((n_states, n_states))
    R_right = R_left.copy()

    R_up = R_left.copy()
    R_down = R_up.copy()

    # If rewarding state is north
    R_up[JUNCTION_UP, TERMINAL] = good_term_reward
    R_down[JUNCTION_UP, TERMINAL] = bad_term_reward

    # If rewarding state is south
    R_up[JUNCTION_DOWN, TERMINAL] = bad_term_reward
    R_down[JUNCTION_DOWN, TERMINAL] = good_term_reward

    R = np.array([R_up, R_down, R_right, R_left])

    # Initialize uniformly at random between north start and south start
    p0 = np.zeros(n_states)
    p0[:2] = 0.5

    # Observation function with have 5 possible obs:
    # start_up, start_down, corridor, junction, and terminal
    phi = np.zeros((n_states, 4 + 1))

    # The two start states have observations of their own
    # (up or down)
    phi[0, 0] = 1
    phi[1, 1] = 1

    # All corridor states share the same observation (idx 2)
    phi[2:-3, 2] = 1

    # All junction states share the same observation (idx 3)
    phi[-3:-1, 3] = 1

    # we have a special termination observations
    phi[-1, 4] = 1

    return T, R, discount, p0, phi

def slippery_tmaze(n: int, discount: float = 0.9, slip_prob: float = 0.1):
    T, R, discount, p0, phi = tmaze(n, discount=discount)

    # First, create a transition matrix w/ a prob of slip_prob of
    # staying in the same state (hence the np.eye call)
    # This is an identity matrix with slip_prob at diagonal
    slip_T = np.eye(T.shape[-1]) * slip_prob

    # We add these slip probabilities to ALL actions for all states.
    # So all actions have a probability slip_prob of resulting in a no-op.
    # Now we need to remove slip_prob probability mass from the rest of the transitions.
    # Transitions sum to 1, so we multiply by (1 - slip_prob) so they sum to the latter.
    T *= (1 - slip_prob)

    # Add our slip probabilities to all actions and all states
    T += slip_T

    return T, R, discount, p0, phi

def four_tmaze(n: int = 1, discount: float = 0.9,
               good_term_reward: float = 4.0,
               bad_term_reward: float = -0.1):

    n_states = 4 * n # Corridor
    n_states += 4 # Start
    n_states += 4 # Second start
    n_states += 4 # Junction
    n_states += 1 # Terminal state (includes both goal/non-goal)

    J_UP_1 = -5
    J_DOWN_1 = -4
    J_UP_2 = -3
    J_DOWN_2 = -2
    junctions = [J_UP_1, J_DOWN_1, J_UP_2, J_DOWN_2]

    S_UP_1 = 0
    S_DOWN_1 = 1
    S_UP_2 = 2
    S_DOWN_2 = 3
    starts = [S_UP_1, S_DOWN_1, S_UP_2, S_DOWN_2]

    SS_UP_1 = 4
    SS_DOWN_1 = 5
    SS_UP_2 = 6
    SS_DOWN_2 = 7
    second_starts = [SS_UP_1, SS_DOWN_1, SS_UP_2, SS_DOWN_2]

    T_up = np.eye(n_states, n_states)
    T_down = T_up.copy()

    zero_self_states = [TERMINAL, J_UP_1, J_UP_2, J_DOWN_1, J_DOWN_2]
    T_up[zero_self_states, zero_self_states] = 0
    T_down[zero_self_states, zero_self_states] = 0

    # If we go up or down at the junctions, we terminate
    terminals = [TERMINAL] * 4
    T_up[junctions, terminals] = 1
    T_down[junctions, terminals] = 1

    T_left = np.zeros((n_states, n_states))
    T_right = T_left.copy()

    # At the leftmost and rightmost states we transition to ourselves
    T_left[starts, starts] = 1
    T_right[junctions, junctions] = 1

    # transition to -2 (left) or +2 (right) index
    all_nonterminal_idxes = np.arange(n_states - 1)
    T_left[all_nonterminal_idxes[4:], all_nonterminal_idxes[:-4]] = 1
    T_right[all_nonterminal_idxes[:-4], all_nonterminal_idxes[4:]] = 1

    T = np.array([T_up, T_down, T_right, T_left])

    # Specify last state as terminal
    T[:, TERMINAL, TERMINAL] = 1

    R_left = np.zeros((n_states, n_states))
    R_right = R_left.copy()

    R_up = R_left.copy()
    R_down = R_up.copy()

    # If rewarding state is north
    R_up[[J_UP_1, J_UP_2], [TERMINAL, TERMINAL]] = good_term_reward
    R_down[[J_UP_1, J_UP_2], [TERMINAL, TERMINAL]] = bad_term_reward

    # If rewarding state is south
    R_up[[J_DOWN_1, J_DOWN_2], [TERMINAL, TERMINAL]] = bad_term_reward
    R_down[[J_DOWN_1, J_DOWN_2], [TERMINAL, TERMINAL]] = good_term_reward

    R = np.array([R_up, R_down, R_right, R_left])

    # Initialize uniformly at random between north start and south start
    p0 = np.zeros(n_states)
    p0[:4] = 1/4

    # Observation function with have 5 possible obs:
    # start_up, start_down, corridor, junction, and terminal
    phi = np.zeros((n_states, 6 + 1))

    # The two start states have observations of their own
    # (up or down)
    phi[[S_UP_1, S_DOWN_1], [0, 0]] = 1
    phi[[S_UP_2, S_DOWN_2], [1, 1]] = 1

    phi[[SS_UP_1, SS_DOWN_2], [2, 2]] = 1
    phi[[SS_DOWN_1, SS_UP_2], [3, 3]] = 1

    # All corridor states share the same observation (idx 2)
    phi[SS_DOWN_2 + 1:J_UP_1, 4] = 1

    # All junction states share the same observation (idx 3)
    phi[junctions, [5] * len(junctions)] = 1

    # we have a special termination observations
    phi[-1, 6] = 1

    return T, R, discount, p0, phi
