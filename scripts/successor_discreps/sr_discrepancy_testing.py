# %%
import numpy as np


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


# %%
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

    if np.allclose(SR_MC, SR_TD):
        times_equal += 1
# %%
# %% Test Markov obs
times_equal = 0
for _ in range(10000):
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

    if np.allclose(SR_MC, SR_TD):
        times_equal += 1
# %%
