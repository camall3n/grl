from grl.mdp import MDP, POMDP

import numpy as np
from jax import jit, nn
import jax.numpy as jnp

@jit
def memory_cross_product(mem_params: jnp.ndarray, amdp: POMDP):
    T_mem = nn.softmax(mem_params, axis=-1)
    n_states_m = T_mem.shape[-1]
    n_states = amdp.state_space.n
    n_states_x = n_states_m * n_states

    # Rewards only depend on MDP (not memory function)
    R_x = amdp.R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SxAxMxM
    T_mem_phi = jnp.tensordot(amdp.phi, T_mem.swapaxes(0, 1), axes=1)

    # Outer product that compacts the two i dimensions and the two l dimensions
    # (SxAxMxM, AxSxS -> AxSMxSM), where SM=x
    T_x = jnp.einsum('iljk,lim->lijmk', T_mem_phi, amdp.T).reshape(amdp.T.shape[0], n_states_x,
                                                                   n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = jnp.kron(amdp.phi, np.eye(n_states_m))

    # Assuming memory starts with all 0s
    p0_x = jnp.zeros(n_states_x)
    p0_x = p0_x.at[::n_states_m].set(amdp.p0)

    mem_aug_mdp = MDP(T_x, R_x, p0_x, gamma=amdp.gamma)
    return POMDP(mem_aug_mdp, phi_x)
