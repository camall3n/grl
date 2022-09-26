from .mdp import MDP, AbstractMDP

# import numpy as np
from jax import jit
import jax.numpy as jnp
from jax.config import config
from functools import partial

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

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

    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(n_states_m, n_states, T, T_mem, phi, R,
                                                            amdp.p0)

    mdp_x = MDP(T_x, R_x, p0_x, amdp.gamma)
    return AbstractMDP(mdp_x, phi_x)

@partial(jit, static_argnums=[0, 1])
def functional_memory_cross_product(n_states_m: int, n_states: int, T: jnp.ndarray,
                                    T_mem: jnp.ndarray, phi: jnp.ndarray, R: jnp.ndarray,
                                    p0: jnp.ndarray):
    n_states_x = n_states_m * n_states
    # T_x = jnp.zeros((T.shape[0], n_states_x, n_states_x))
    #
    #
    # R_x = jnp.zeros((R.shape[0], n_states_x, n_states_x))
    # phi_x = jnp.zeros((n_states_x, phi.shape[-1] * n_states_m))

    # Rewards only depend on MDP (not memory function)
    R_x = R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SMxM
    T_mem_phi = jnp.tensordot(phi, T_mem, axes=1)

    # Outer product that compacts the 2 i
    T_x = jnp.einsum('ijk,lim->lijmk', T_mem_phi, T).reshape(T.shape[0], n_states_x, n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = jnp.kron(phi, jnp.eye(n_states_m))

    # Assuming memory starts with all 0s
    p0_x = jnp.zeros(n_states_x)
    p0_x = p0_x.at[::n_states_m].set(p0)

    return T_x, R_x, p0_x, phi_x
