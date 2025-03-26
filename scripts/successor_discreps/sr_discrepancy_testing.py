# %%
import numpy as np
np.set_printoptions(precision=3, suppress=True)


def dot(x, *args):
    while args:
        y, args = args[0], args[1:]
        x = np.tensordot(x, y, axes=1)
    return x


def ddot(x, *args):
    while args:
        y, args = args[0], args[1:]
        # x = np.einsum('ijkl,klmn->ijmn', x, y)
        x = np.tensordot(x, y, axes=2)
    return x


def dpow(a, exp):
    x, *rest = [a] * exp
    return ddot(x, *rest)


def kron(a, b):
    return np.einsum("ij...,kl...->ikjl...", a, b)

# %% Setup environments

from grl.mdp import MDP, POMDP
from grl.environment.tmaze import *

def setup_tmaze(up_prob: float = .5, gamma: float = 1.0):
    corridor_length = 4
    T, R, _, p0, phi = tmaze(corridor_length, discount=gamma)

    # adjust T to be a sub-probability matrix where the remaining mass means termination
    #STATE_TERMINAL = -1
    #T[:, STATE_TERMINAL, :] = 0.0

    # instantiate mdp
    mdp = MDP(T, R, p0, gamma=gamma)
    env = POMDP(mdp, phi)


    # policy
    OBS_UP = 0
    OBS_DOWN = 1
    OBS_CORRIDOR = 2
    OBS_JUNCTION = 3
    OBS_TERMINAL = 4

    A_UP = 0
    A_DOWN = 1
    A_RIGHT = 2
    A_LEFT = 3

    n_actions = env.action_space.n
    n_obs = env.observation_space.n

    assert up_prob >= 0.0 and up_prob <= 1.0

    pi = np.zeros((n_obs, n_actions), dtype=float)
    pi[[OBS_UP, OBS_DOWN, OBS_CORRIDOR], A_RIGHT] = 1.0
    pi[OBS_JUNCTION, A_UP] = up_prob
    pi[OBS_JUNCTION, A_DOWN] = 1.0 - up_prob
    pi[OBS_TERMINAL, A_RIGHT] = 1.0  # doesn't matter
    assert pi.shape == (n_obs, n_actions)
    assert np.allclose(pi.sum(axis=1), 1.0)

    return env, pi

def setup_tmaze_two_goals(up_prob: float = .5, gamma: float = 1.0):
    corridor_length = 4
    T, R, gamma, p0, phi = tmaze_two_goals(corridor_length, discount=gamma)

    # adjust T to be a sub-probability matrix where the remaining mass means termination
    #STATE_GOAL = -2
    #STATE_TERMINAL = -1
    #T[:, [STATE_GOAL, STATE_TERMINAL], :] = 0.0

    # instantiate environment
    mdp = MDP(T, R, p0, gamma=gamma)
    env = POMDP(mdp, phi)

    # policy
    OBS_UP = 0
    OBS_DOWN = 1
    OBS_CORRIDOR = 2
    OBS_JUNCTION = 3
    OBS_GOAL = 4
    OBS_TERMINAL = 5

    A_UP = 0
    A_DOWN = 1
    A_RIGHT = 2
    A_LEFT = 3

    n_actions = env.action_space.n
    n_states = env.state_space.n
    n_obs = env.observation_space.n

    assert up_prob >= 0 and up_prob <= 1
    pi = np.zeros((n_obs, n_actions), dtype=float)
    pi[[OBS_UP, OBS_DOWN, OBS_CORRIDOR], A_RIGHT] = 1.0
    pi[OBS_JUNCTION, A_UP] = up_prob
    pi[OBS_JUNCTION, A_DOWN] = 1.0 - up_prob
    pi[[OBS_GOAL, OBS_TERMINAL], A_RIGHT] = 1.0  # doesn't matter
    assert pi.shape == (n_obs, n_actions)
    assert np.allclose(pi.sum(axis=1), 1.0)

    return env, pi

from grl.environment import load_pomdp

def setup_parity_check(up_prob: float = .5, gamma: float = 1.0, reward_in_obs: bool=True):
    env, _ = load_pomdp("parity_check", reward_in_obs=reward_in_obs)
    env.gamma = gamma
    """ the four distinct corridors 
    `colour1 -> colour2 -> junction` are (1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12), and the final state is 13
    with folded in rewards we have instead
    1 - 5 - 9  -> {15, 13} 
    2 - 6 - 10 -> {13, 15}
    3 - 7 - 11 -> {15, 13}
    4 - 8 - 12 -> {13, 15}
    """

    # adjust T to be a sub-probability matrix where the remaining mass means termination
    #if reward_in_obs:
    #    # 13 is actually never reached but we have to zero it anyway
    #    TERMINAL_STATES = [12, 13, 14]
    #else:
    #    TERMINAL_STATES = [12]
    #env.T[:, TERMINAL_STATES, :] = 0.0

    # policy
    # there are only two actions, up and down, and they don't make a difference except in the junction state
    A_UP = 0
    A_DOWN = 1

    n_actions = env.action_space.n
    n_states = env.state_space.n
    n_obs = env.observation_space.n

    assert up_prob >= 0 and up_prob <= 1
    pi = np.zeros((n_obs, n_actions), dtype=float)
    pi[:, A_UP] = up_prob
    pi[:, A_DOWN] = 1.0 - up_prob
    assert pi.shape == (n_obs, n_actions)
    assert np.allclose(pi.sum(axis=1), 1.0)

    return env, pi

#%% sr discrepancy code

def is_prob_matrix(P, shape = None):
    if shape is not None and P.shape != shape:
        return False
    return np.all(P >= 0) and np.allclose(P.sum(axis=-1), 1.0)

def is_subprob_matrix(P, shape = None):
    epsilon = 1e-7
    if shape is not None and P.shape != shape:
        return False
    return np.all(P >= -epsilon) and np.all(P.sum(axis=-1) <= 1.0 + epsilon)

def make_subprob_matrix(T):
    """Find terminal states, which are those where every action just leads back to 
    the same state, and set their out-probabilities to zero, which means that we actually
    terminate after that state; this makes it so that geometric sum formulas with gamma=1
    give you the correct answer"""
    
    a, b, c = T.shape
    if b==c and a!=b:
        n_actions = a
        n_states = b
        T = np.permute_dims(T, (1, 0, 2))
    elif a==c and a!=b:
        n_actions = b
        n_states = a
    else:
        assert False, "shape of T has to be (S,A,S) or (A,S,S) and A!=S, otherwise we cant tell which one it is"
    
    # now T.shape = (S, A, S)

    # find terminal states
    def is_terminal(s):
        return np.allclose(T[s, :, s], 1.0) \
            and np.allclose(T[s, :, :s], 0.0) \
            and np.allclose(T[s, :, (s+1):], 0.0)

    Tnew = np.copy(T)
    for s in range(n_states):
        if is_terminal(s):
            Tnew[s, :, :] = 0
    return Tnew

def calculate_sr_discrepancy_from_env(env: POMDP, pi: np.ndarray):
    n_actions = env.action_space.n
    n_states = env.state_space.n
    n_obs = env.observation_space.n

    Phi = env.phi
    T = env.T
    p0 = env.p0
    gamma = env.gamma
    return calculate_sr_discrepancy_raw(
        n_actions,
        n_states,
        n_obs,
        Phi,
        T,
        p0,
        gamma,
        pi
    )

def calculate_sr_discrepancy_raw(
        n_actions: int,
        n_states: int,
        n_obs: int,
        Phi: np.ndarray,
        T: np.ndarray,
        p0: np.ndarray,
        gamma: float,
        pi: np.ndarray,
    ):
    # observation matrix
    assert is_prob_matrix(Phi, (n_states, n_obs))

    # transition function
    assert n_actions != n_states, f"n_actions = n_states = {n_actions}, which means we can't tell if T has shape (S,A,S) or (A,S,S)"
    if T.shape == (n_actions, n_states, n_states):
        T = np.permute_dims(T, (1, 0, 2))
    assert is_subprob_matrix(T, (n_states, n_actions, n_states))
    T = make_subprob_matrix(T)
    assert is_subprob_matrix(T, (n_states, n_actions, n_states))

    # initial state distribution
    assert is_prob_matrix(p0, (n_states,))

    I_S = np.eye(n_states)
    I_A = np.eye(n_actions)
    I_O = np.eye(n_obs)
    I_SA = np.eye(n_actions * n_states).reshape((n_states, n_actions, n_states, n_actions))
    Phi_A = kron(Phi, I_A)

    
    pi_s = dot(Phi, pi)
    assert is_prob_matrix(pi_s, (n_states, n_actions))
    T_pi = np.einsum("ik,ikj->ij", pi_s, T)
    assert is_subprob_matrix(T_pi, (n_states, n_states))
    Pi = np.eye(len(pi))[..., None] * pi[None, ...]
    Pi_s = np.eye(len(pi_s))[..., None] * pi_s[None, ...]

    Pr_s = np.ones(n_states) / n_states
    Pr_s = np.linalg.inv(I_S - gamma * T_pi.T).dot(p0)
    #Pr_s = np.random.random(n_states)
    Pr_s = Pr_s / np.sum(Pr_s)
    #Pr_s[0] = (Pr_s[0] + Pr_s[1] + Pr_s[2] + Pr_s[3]) / 4
    #Pr_s[1] = Pr_s[0]
    #Pr_s[2] = Pr_s[0]
    #Pr_s[3] = Pr_s[0]
    W = np.zeros((n_obs, n_states))
    for i in range(n_obs):
        for j in range(n_states):
            pr_i = np.sum([Pr_s[k] * Phi[k][i] for k in range(n_states)])
            if np.isclose(pr_i, 0.0):
                # this observation is never dispensed...
                continue
            W[i,j] = (
                Pr_s[j] * Phi[j][i] / pr_i
            )

    W_Pi = ddot(Pi, kron(W, I_A))

    SR_MC_SS = np.linalg.inv(I_S - gamma * ddot(Pi_s, T))
    SR_TD_SS = np.linalg.inv(I_S - gamma * ddot(dot(Phi, W_Pi), T))

    """
    print(f"SR_MC_SS = SR_TD_SS? {np.allclose(SR_MC_SS, SR_TD_SS)}")

    A_MC = dot(SR_MC_SS, Phi)
    A_TD = dot(SR_TD_SS, Phi)

    print(f"X Phi equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(ddot(W_Pi, T), SR_MC_SS)
    A_TD = dot(ddot(W_Pi, T), SR_TD_SS)

    print(f"WPi T X equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(dot(T, SR_MC_SS), Phi)
    A_TD = dot(dot(T, SR_TD_SS), Phi)

    print(f"T X Phi equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(dot(ddot(kron(W, I_A), T), SR_MC_SS), Phi)
    A_TD = dot(dot(ddot(kron(W, I_A), T), SR_TD_SS), Phi)

    print(f"W T X Phi equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(dot(ddot(W_Pi, T), SR_MC_SS), Phi)
    A_TD = dot(dot(ddot(W_Pi, T), SR_TD_SS), Phi)

    print(f"WPi T X Phi equal? {np.allclose(A_MC, A_TD)}")
    """

    SR_MC = I_O + gamma * dot(ddot(W_Pi, T), SR_MC_SS, Phi)
    SR_TD = I_O + gamma * dot(ddot(W_Pi, T), SR_TD_SS, Phi)
    #SR_TD = np.linalg.inv(I_O - gamma * dot(ddot(W_Pi, T), Phi))

    return SR_MC, SR_TD

# %% 

if __name__ == "__main__":
    gamma = 1.0
    envs = {
        #"Two-goal T-Maze (up-probability 2/3)": setup_tmaze_two_goals(2 / 3, gamma=gamma),
        #"Two-goal T-Maze (up-probability 1/2)": setup_tmaze_two_goals(1 / 2, gamma=gamma),
        #"Ordinary T-Maze (up-probability 2/3)": setup_tmaze(2 / 3, gamma=gamma),
        #"Ordinary T-Maze (up-probability 1/2)": setup_tmaze(1 / 2, gamma=gamma),
        "Parity check (up-probability 1/2, reward not included)":  setup_parity_check(1/2, gamma=gamma, reward_in_obs=False),
        "Parity check (up-probability 1/2, reward is  included)":  setup_parity_check(1/2, gamma=gamma, reward_in_obs=True),
        "Parity check (up-probability 2/3, reward not included)":  setup_parity_check(2/3, gamma=gamma, reward_in_obs=False),
        "Parity check (up-probability 2/3, reward is  included)":  setup_parity_check(2/3, gamma=gamma, reward_in_obs=True),
    }

    for name, x in envs.items():
        mc, td = calculate_sr_discrepancy_from_env(*x)
        discrepancy = np.sum(np.abs(mc - td))
        print(f"discrepancy for {name}: {discrepancy:.3f}")
        #print(mc)
        #print(td)
        if not np.isclose(discrepancy, 0.0):
            print(mc - td)


# %% non-trivial observations

if __name__ == "__main__":
    times_equal = 0
    for _ in range(10000):
        A = 2
        S = 4
        O = 3

        gamma = 0.9

        pi = np.random.rand(O, A)
        pi = pi / pi.sum(axis=-1)[:, None]

        Phi = np.random.rand(S, O)
        Phi = Phi / Phi.sum(axis=-1)[:, None]

        T = np.random.rand(S, A, S)
        T = T / T.sum(axis=2)[:, :, None]

        s0 = np.random.rand(S)
        s0 = s0 / s0.sum()

        I_S = np.eye(S)
        I_A = np.eye(A)
        I_O = np.eye(O)
        I_SA = np.eye(A * S).reshape((S, A, S, A))
        Phi_A = kron(Phi, I_A)

        pi_s = dot(Phi, pi)
        T_pi = np.einsum("ik,ikj->ij", pi_s, T)
        Pi = np.eye(len(pi))[..., None] * pi[None, ...]
        Pi_s = np.eye(len(pi_s))[..., None] * pi_s[None, ...]

        Pr_s = np.linalg.inv(I_S - gamma * T_pi.T).dot(s0)
        Pr_s = Pr_s / np.sum(Pr_s)
        W = np.zeros((O, S))
        for i in range(O):
            for j in range(S):
                W[i][j] = (
                    Pr_s[j] * Phi[j][i] / np.sum([Pr_s[k] * Phi[k][i] for k in range(S)])
                )

        W_Pi = ddot(Pi, kron(W, I_A))

        SR_MC_SS = np.linalg.inv(I_S - gamma * ddot(Pi_s, T))
        SR_TD_SS = np.linalg.inv(I_S - gamma * ddot(dot(Phi, W_Pi), T))

        SR_MC = I_O + gamma * dot(ddot(W_Pi, T), SR_MC_SS, Phi)
        SR_TD = np.linalg.inv(I_O - gamma * dot(ddot(W_Pi, T), Phi))

        SR_MC, SR_TD = calculate_sr_discrepancy_raw(A, S, O, Phi, T, s0, gamma, pi)

        if np.allclose(SR_MC, SR_TD):
            times_equal += 1

    assert times_equal == 0


# %% Test Markov obs

    times_equal = 0
    for _ in range(1000):
        A = 2
        S = 4
        O = 4

        gamma = 0.9

        pi = np.random.rand(O, A)
        pi = pi / pi.sum(axis=-1)[:, None]

        # we want a permutation of an identity matrix to verify that it's non-trivially working
        while True:
            Phi = np.random.permutation(np.eye(O))
            if not np.allclose(Phi, np.eye(O)):
                break

        T = np.random.rand(S, A, S)
        T = T / T.sum(axis=2)[:, :, None]

        s0 = np.random.rand(S)
        s0 = s0 / s0.sum()

        I_S = np.eye(S)
        I_A = np.eye(A)
        I_O = np.eye(O)
        I_SA = np.eye(A * S).reshape((S, A, S, A))
        Phi_A = kron(Phi, I_A)

        pi_s = dot(Phi, pi)
        T_pi = np.einsum("ik,ikj->ij", pi_s, T)
        Pi = np.eye(len(pi))[..., None] * pi[None, ...]
        Pi_s = np.eye(len(pi_s))[..., None] * pi_s[None, ...]

        Pr_s = np.linalg.inv(I_S - gamma * T_pi.T).dot(s0)
        Pr_s = Pr_s / np.sum(Pr_s)
        W = np.zeros((O, S))
        for i in range(O):
            for j in range(S):
                W[i][j] = (
                    Pr_s[j] * Phi[j][i] / np.sum([Pr_s[k] * Phi[k][i] for k in range(S)])
                )

        W_Pi = ddot(Pi, kron(W, I_A))

        SR_MC_SS = np.linalg.inv(I_S - gamma * ddot(Pi_s, T))
        SR_TD_SS = np.linalg.inv(I_S - gamma * ddot(dot(Phi, W_Pi), T))

        SR_MC = I_O + gamma * dot(ddot(W_Pi, T), SR_MC_SS, Phi)
        SR_TD = np.linalg.inv(I_O - gamma * dot(ddot(W_Pi, T), Phi))

        SR_MC, SR_TD = calculate_sr_discrepancy_raw(A, S, O, Phi, T, s0, gamma, pi)

        if np.allclose(SR_MC, SR_TD):
            times_equal += 1

    assert times_equal == 1000