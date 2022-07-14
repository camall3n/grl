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
    n_states_m = T_mem.shape[-1]
    n_states_x = T_mem.shape[-1] * T.shape[-1] # cross (x) product MDP
    T_x = np.zeros((T.shape[0], n_states_x, n_states_x))
    R_x = np.zeros((R.shape[0], n_states_x, n_states_x))
    phi_x = np.zeros((n_states_x, phi.shape[-1] * T_mem.shape[-1]))

    for s in range(amdp.n_states):
        # Need to skip getting next_state for terminal states
        terminal = False
        if np.all(T[:, s] == 0):
            terminal = True

        T_phi_mem = np.tensordot(phi[s][:-1], T_mem, 1) # T_mem does not include terminal state
        for a in range(T.shape[0]):
            if not terminal:
                rows = np.multiply.outer(T[a, s],
                                         T_phi_mem).swapaxes(0, 1).reshape(n_states_m, n_states_x)
                T_x[a, (s * n_states_m):(s * n_states_m + n_states_m)] = rows

                # Rewards only depend on MDP (not memory function)
                R_x[a, s * n_states_m:s * n_states_m + n_states_m] = np.repeat(R[a, s], n_states_m)

            for s_mem in range(n_states_m):
                phi_x[s * 2 + s_mem][s_mem::n_states_m] = phi[s]

    mdp = MDP(T_x, R_x, amdp.gamma)
    # Assuming memory starts with all 0s
    p0_x = np.zeros(T_x.shape[-1])
    p0_x[::n_states_m] = amdp.p0
    return AbstractMDP(mdp, phi_x, p0=p0_x)
