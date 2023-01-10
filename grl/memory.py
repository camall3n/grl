from .mdp import MDP, AbstractMDP

import numpy as np
from jax import jit, nn
import jax.numpy as jnp
from tqdm import tqdm

def memory_cross_product(amdp, mem_params: jnp.ndarray):
    """
    Returns AMDP resulting from cross product of the underlying MDP with given memory function

    :param amdp:  AMDP
    :param mem_params: memory transition function parameters
    """
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(amdp.T, T_mem, amdp.phi, amdp.R,
                                                            amdp.p0)

    mdp_x = MDP(T_x, R_x, p0_x, amdp.gamma)
    return AbstractMDP(mdp_x, phi_x)

@jit
def functional_memory_cross_product(T: jnp.ndarray, T_mem: jnp.ndarray, phi: jnp.ndarray,
                                    R: jnp.ndarray, p0: jnp.ndarray):
    n_states_m = T_mem.shape[-1]
    n_states = T.shape[-1]
    n_states_x = n_states_m * n_states

    # Rewards only depend on MDP (not memory function)
    R_x = R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SxAxMxM
    T_mem_phi = jnp.tensordot(phi, T_mem.swapaxes(0, 1), axes=1)

    # Outer product that compacts the two i dimensions and the two l dimensions
    # (SxAxMxM, AxSxS -> AxSMxSM), where SM=x
    T_x = jnp.einsum('iljk,lim->lijmk', T_mem_phi, T).reshape(T.shape[0], n_states_x, n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = jnp.kron(phi, np.eye(n_states_m))

    # Assuming memory starts with all 0s
    p0_x = jnp.zeros(n_states_x)
    p0_x = p0_x.at[::n_states_m].set(p0)

    return T_x, R_x, p0_x, phi_x

def generate_1bit_mem_fns(n_obs, n_actions):
    """
    Generates all possible deterministic 1 bit memory functions with given number of obs and actions.
    There are M^(MZA) memory functions.
    1 bit means M=2.

    Example:
    For 2 obs (r, b) and 2 actions (up, down), binary_mp=10011000 looks like:

    m o_a    mp
    -----------
    0 r_up   1
    0 r_down 0
    0 b_up   0
    0 b_down 1
    1 r_up   1
    1 r_down 0
    1 b_up   0
    1 b_down 0

    """
    # TODO: add tests
    n_mem_states = 2
    fns = []

    MZA = n_mem_states * n_obs * n_actions
    for i in tqdm(range(n_mem_states**(MZA))):
        T_mem = generate_mem_fn(i, n_mem_states, n_obs, n_actions)
        fns.append(T_mem)

    return fns

def generate_mem_fn(mem_fn_id, n_mem_states, n_obs, n_actions):
    """Generate the AxZxMxM memory function transition matrix for the given
    mem_fn_id and sizes.

    :param mem_fn_id: a decimal number whose binary representation is m'
    """

    MZA = n_mem_states * n_obs * n_actions
    n_valid_mem_fns = n_mem_states**MZA
    if mem_fn_id is not None and (mem_fn_id >= n_valid_mem_fns or mem_fn_id < 0):
        raise ValueError(f'Unknown mem_fn_id: {mem_fn_id}')

    binary_mp = format(mem_fn_id, 'b').zfill(MZA)
    T_mem = np.zeros((n_actions, n_obs, n_mem_states, n_mem_states))
    for m in range(n_mem_states):
        for ob in range(n_obs):
            for a in range(n_actions):
                mp = int(binary_mp[m * n_obs * n_actions + ob * n_actions + a])
                T_mem[a, ob, m, mp] = 1
    return T_mem
