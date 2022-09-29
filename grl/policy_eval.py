import logging

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

from .mdp import MDP
from .memory import memory_cross_product

class PolicyEval:
    def __init__(self, amdp, verbose=True):
        """
        :param amdp:     AMDP
        :param verbose:  log everything
        """
        self.amdp = amdp
        self.verbose = verbose

    def run(self, pi_abs):
        """
        :param pi_abs: policy to evaluate, defined over abstract state space
        """
        return self._functional_run(pi_abs, self.amdp.phi, self.amdp.T, self.amdp.R, self.amdp.p0,
                                    self.amdp.gamma, self.amdp.n_states, self.amdp.n_obs)

    @partial(jit, static_argnames=['self', 'gamma', 'n_states', 'n_obs'])
    def _functional_run(self, pi_abs: jnp.ndarray, phi: jnp.ndarray, T: jnp.ndarray,
                        R: jnp.ndarray, p0: jnp.ndarray, gamma: float, n_states: int, n_obs: int):
        pi_ground = phi @ pi_abs

        # MC*
        state_v, state_q = self._functional_solve_mdp(pi_ground, T, R, gamma, n_states)
        state_vals = {'v': state_v, 'q': state_q}

        occupancy = self._functional_get_occupancy(pi_ground, T, p0, n_states, gamma)

        p_pi_of_s_given_o = self._get_p_s_given_o(n_obs, phi, occupancy)
        mc_vals = self._functional_solve_amdp(state_q, p_pi_of_s_given_o, pi_abs)

        # TD
        T_obs_obs, R_obs_obs = self._functional_create_td_model(p_pi_of_s_given_o, phi, T, R,
                                                                n_obs)
        td_v_vals, td_q_vals = self._functional_solve_mdp(pi_abs, T_obs_obs, R_obs_obs, gamma,
                                                          T_obs_obs.shape[-1])
        td_vals = {'v': td_v_vals, 'q': td_q_vals}

        return state_vals, mc_vals, td_vals

    def _solve_mdp(self, mdp, pi):
        v, q = self._functional_solve_mdp(pi, mdp.T, mdp.R, mdp.gamma, mdp.n_states)

        return {'v': v, 'q': q}

    @staticmethod
    @partial(jit, static_argnums=[0])
    def _get_p_s_given_o(n_obs: int, phi: jnp.ndarray, occupancy: jnp.ndarray):
        repeat_occupancy = jnp.repeat(occupancy[..., None], n_obs, -1)

        # Q vals
        p_of_o_given_s = phi.astype('float')
        w = repeat_occupancy * p_of_o_given_s

        p_pi_of_s_given_o = w / (w.sum(axis=0) + 1e-10)
        return p_pi_of_s_given_o

    @staticmethod
    @partial(jit, static_argnames=['gamma', 'n_states'])
    def _functional_solve_mdp(pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, gamma: float,
                              n_states: int):
        """
        Solves for V using linear equations.
        For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
        """
        Pi_pi = pi.transpose()[..., None]
        T_pi = (Pi_pi * T).sum(axis=0) # T^π(s'|s)
        R_pi = (Pi_pi * T * R).sum(axis=0).sum(axis=-1) # R^π(s)

        # A*V_pi(s) = b
        # A = (I - \gamma (T^π))
        # b = R^π
        A = (jnp.eye(n_states) - gamma * T_pi)
        b = R_pi
        v_vals = jnp.linalg.solve(A, b)

        R_sa = (T * R).sum(axis=-1) # R(s,a)
        q_vals = (R_sa + (gamma * T @ v_vals))

        return v_vals, q_vals

    def get_occupancy(self, pi: jnp.ndarray):
        """
        Finds the visitation count, C_pi(s), of each state.
        For all s, C_pi(s) = p0(s) + sum_s^ sum_a[C_pi(s^) * gamma * T(s|a,s^) * pi(a|s^)],
          where s^ is the prev state
        """
        pi_ground = self.amdp.phi @ pi
        return self._functional_get_occupancy(pi_ground, self.amdp.T, self.amdp.p0,
                                              self.amdp.n_states, self.amdp.gamma)

    @staticmethod
    @partial(jit, static_argnames=['n_states', 'gamma'])
    def _functional_get_occupancy(pi_ground: jnp.ndarray, T: jnp.ndarray, p0: jnp.ndarray,
                                  n_states: int, gamma: float):
        Pi_pi = pi_ground.transpose()[..., None]
        T_pi = (Pi_pi * T).sum(axis=0) # T^π(s'|s)

        # A*C_pi(s) = b
        # A = (I - \gamma (T^π)^T)
        # b = P_0
        A = jnp.eye(n_states) - gamma * T_pi.transpose()
        b = p0
        return jnp.linalg.solve(A, b)

    def _solve_amdp(self, mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                    pi: jnp.ndarray):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        return self._functional_solve_amdp(mdp_q_vals, p_pi_of_s_given_o, pi)

    @partial(jit, static_argnums=0)
    def _functional_solve_amdp(self, mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                               pi_abs: jnp.ndarray):

        # Q vals
        amdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

        # V vals
        amdp_v_vals = (amdp_q_vals * pi_abs.T).sum(0)

        return {'v': amdp_v_vals, 'q': amdp_q_vals}

    def _create_td_model(self, p_pi_of_s_given_o: jnp.ndarray):
        T_obs_obs, R_obs_obs = self._functional_create_td_model(p_pi_of_s_given_o, self.amdp.phi,
                                                                self.amdp.T, self.amdp.R,
                                                                self.amdp.n_obs)
        return MDP(T_obs_obs, R_obs_obs, self.amdp.p0, self.amdp.gamma)

    @staticmethod
    @partial(jit, static_argnames=['n_obs'])
    def _functional_create_td_model(p_pi_of_s_given_o: jnp.ndarray, phi: jnp.ndarray,
                                    T: jnp.ndarray, R: jnp.ndarray, n_obs: int):
        # creates an (n_obs * n_obs) x 2 array of all possible observation to observation pairs.
        # we flip here so that we have curr_obs, next_obs (order matters).
        obs_idx_product = jnp.flip(
            jnp.dstack(jnp.meshgrid(jnp.arange(n_obs), jnp.arange(n_obs))).reshape(-1, 2), -1)

        # this gives us (n_obs * n_obs) x states x 1 and (n_obs * n_obs) x 1 x states
        curr_s_given_o = p_pi_of_s_given_o[:, obs_idx_product[:, 0]].T[..., None]
        next_o_given_s = jnp.expand_dims(phi[:, obs_idx_product[:, 1]].T, 1)

        # outer product here
        o_to_next_o = jnp.expand_dims(curr_s_given_o * next_o_given_s, 1)

        # This is p(o, s, a, s', o')
        # the probability that o goes to o', via each path (s, a) -> s'.
        # Shape is (n_obs * n_obs) x |A| x |S| x |S|
        T_contributions = T * o_to_next_o

        # |A| x (n_obs * n_obs)
        T_obs_obs_flat = T_contributions.sum(-1).sum(-1).T

        # |A| x n_obs x n_obs
        T_obs_obs = T_obs_obs_flat.reshape(T.shape[0], n_obs, n_obs)

        # You want everything to sum to one
        denom = T_obs_obs_flat.T[..., None, None]
        denom_no_zero = denom + (denom == 0).astype(denom.dtype)

        R_contributions = (R * T_contributions) / denom_no_zero
        R_obs_obs_flat = R_contributions.sum(-1).sum(-1).T
        R_obs_obs = R_obs_obs_flat.reshape(R.shape[0], n_obs, n_obs)

        return T_obs_obs, R_obs_obs

    ##########
    # Helpers for gradient/heatmap stuff
    ##########

    def mse_loss(self, pi, value_type, **kwargs):
        """
        sum_o [V_td^pi(o) - V_mc^pi(o)]^2
        """
        _, mc_vals, td_vals = self.run(pi)
        diff = mc_vals[value_type] - td_vals[value_type]
        return (diff**2).mean()

    def max_loss(self, pi, value_type, **kwargs):
        """
        max_o abs[V_td^pi(o) - V_mc^pi(o)]
        """
        _, mc_vals, td_vals = self.run(pi)
        return np.abs(mc_vals[value_type] - td_vals[value_type]).max()

    def memory_loss(self, T_mem, value_type, **kwargs):
        amdp = memory_cross_product(self.amdp, T_mem)
        pe = PolicyEval(amdp, verbose=False)
        return pe.mse_loss(kwargs['pi_abs'], value_type)
