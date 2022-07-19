import numpy as np

from .mdp import MDP, AbstractMDP

def memory_cross_product(amdp, T_mem):
    """
    Returns AMDP resulting from cross product of the underlying MDP with given memory function

    :param amdp:  AMDP
    :param T_mem: memory transition function
    """
    T = amdp.T
    R = amdp.R
    phi = amdp.phi
    n_states = T.shape[-1]
    n_states_m = T_mem.shape[-1]
    n_states_x = n_states * n_states_m # cross (x) product MDP
    T_x = np.zeros((T.shape[0], n_states_x, n_states_x))
    R_x = np.zeros((R.shape[0], n_states_x, n_states_x))
    phi_x = np.zeros((n_states_x, phi.shape[-1] * n_states_m))

    # Rewards only depend on MDP (not memory function)
    R_x = R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    T_phi_mem = np.tensordot(phi, T_mem, axes=1)
    T_x_big = np.multiply.outer(T, T_phi_mem).swapaxes(2, 3).swapaxes(3, 4).reshape(
        T.shape[0], n_states, n_states_x, n_states_x)

    for s in range(amdp.n_states):
        start = s * n_states_m
        end = s * n_states_m + n_states_m
        T_x[:, start:end] = T_x_big[:, s, start:end]

        for s_mem in range(n_states_m):
            # The new obs_x are the original obs times memory states
            # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
            phi_x[s * n_states_m + s_mem, s_mem::n_states_m] = phi[s]

    mdp_x = MDP(T_x, R_x, amdp.gamma)
    # Assuming memory starts with all 0s
    p0_x = np.zeros(n_states_x)
    p0_x[::n_states_m] = amdp.p0
    return AbstractMDP(mdp_x, phi_x, p0=p0_x)
