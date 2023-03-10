from jax import jit
import jax.numpy as jnp
from typing import Union
from grl.mdp import MDP, AbstractMDP

@jit
def functional_get_occupancy(pi_ground: jnp.ndarray, mdp: Union[MDP, AbstractMDP]):
    Pi_pi = pi_ground.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)

    # A*C_pi(s) = b
    # A = (I - \gamma (T^π)^T)
    # b = P_0
    A = jnp.eye(mdp.T.shape[-1]) - mdp.gamma * T_pi.transpose()
    b = mdp.p0
    return jnp.linalg.solve(A, b)

def amdp_get_occupancy(pi: jnp.ndarray, amdp: AbstractMDP):
    pi_ground = amdp.phi @ pi
    return functional_get_occupancy(pi_ground, amdp)

@jit
def get_p_s_given_o(phi: jnp.ndarray, occupancy: jnp.ndarray):
    repeat_occupancy = jnp.repeat(occupancy[..., None], phi.shape[-1], -1)

    # Q vals
    p_of_o_given_s = phi.astype(float)
    w = repeat_occupancy * p_of_o_given_s

    p_pi_of_s_given_o = w / (w.sum(axis=0) + 1e-10)
    return p_pi_of_s_given_o

@jit
def functional_create_td_model(p_pi_of_s_given_o: jnp.ndarray, amdp: AbstractMDP):
    # creates an (n_obs * n_obs) x 2 array of all possible observation to observation pairs.
    # we flip here so that we have curr_obs, next_obs (order matters).
    obs_idx_product = jnp.flip(
        jnp.dstack(jnp.meshgrid(jnp.arange(amdp.phi.shape[-1]),
                                jnp.arange(amdp.phi.shape[-1]))).reshape(-1, 2), -1)

    # this gives us (n_obs * n_obs) x states x 1 and (n_obs * n_obs) x 1 x states
    curr_s_given_o = p_pi_of_s_given_o[:, obs_idx_product[:, 0]].T[..., None]
    next_o_given_s = jnp.expand_dims(amdp.phi[:, obs_idx_product[:, 1]].T, 1)

    # outer product here
    o_to_next_o = jnp.expand_dims(curr_s_given_o * next_o_given_s, 1)

    # This is p(o, s, a, s', o')
    # the probability that o goes to o', via each path (s, a) -> s'.
    # Shape is (n_obs * n_obs) x |A| x |S| x |S|
    T_contributions = amdp.T * o_to_next_o

    # |A| x (n_obs * n_obs)
    T_obs_obs_flat = T_contributions.sum(-1).sum(-1).T

    # |A| x n_obs x n_obs
    T_obs_obs = T_obs_obs_flat.reshape(amdp.T.shape[0], amdp.phi.shape[-1], amdp.phi.shape[-1])

    # You want everything to sum to one
    denom = T_obs_obs_flat.T[..., None, None]
    denom_no_zero = denom + (denom == 0).astype(denom.dtype)

    R_contributions = (amdp.R * T_contributions) / denom_no_zero
    R_obs_obs_flat = R_contributions.sum(-1).sum(-1).T
    R_obs_obs = R_obs_obs_flat.reshape(amdp.R.shape[0], amdp.phi.shape[-1], amdp.phi.shape[-1])

    return T_obs_obs, R_obs_obs

def to_dict(T, R, gamma, p0, phi, Pi_phi, Pi_phi_x=None):
    return {
        'T': T,
        'R': R,
        'gamma': gamma,
        'p0': p0,
        'phi': phi,
        'Pi_phi': Pi_phi,
        'Pi_phi_x': Pi_phi_x,
    }

def get_perf(info: dict):
    return (info['state_vals_v'] * info['p0']).sum()
