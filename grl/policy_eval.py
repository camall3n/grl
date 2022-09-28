import logging

import numpy as np
import jax.numpy as jnp
from jax import jit
from itertools import product
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
        self.obs_idx_product = np.array(list(prod for prod in product(np.arange(self.amdp.n_obs), np.arange(self.amdp.n_obs))))

    def run(self, pi_abs):
        """
        :param pi_abs: policy to evaluate, defined over abstract state space
        """
        self.pi_abs = pi_abs
        self.pi_ground = self.amdp.get_ground_policy(pi_abs)

        # MC*
        mdp_vals = self._solve_mdp(self.amdp, self.pi_ground)
        occupancy = self._get_occupancy(self.pi_ground)

        p_pi_of_s_given_o = self._get_p_s_given_o(self.amdp.n_obs, self.amdp.phi, occupancy)
        mc_vals = self._solve_amdp(mdp_vals['q'], p_pi_of_s_given_o)

        if self.verbose:
            logging.info(f'occupancy:\n {occupancy}')

        # TD
        td_vals = self._solve_mdp(self._create_td_model(p_pi_of_s_given_o), self.pi_abs)

        return mdp_vals, mc_vals, td_vals

    def _solve_mdp(self, mdp, pi):
        return self._functional_solve_mdp(pi, mdp.T, mdp.R, mdp.gamma, mdp.n_states)

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

        return {'v': v_vals, 'q': q_vals}

    @partial(jit, static_argnums=0)
    def _get_occupancy(self, pi_ground: jnp.ndarray):
        """
        Finds the visitation count, C_pi(s), of each state.
        For all s, C_pi(s) = p0(s) + sum_s^ sum_a[C_pi(s^) * gamma * T(s|a,s^) * pi(a|s^)],
          where s^ is the prev state
        """
        Pi_pi = pi_ground.transpose()[..., None]
        T_pi = (Pi_pi * self.amdp.T).sum(axis=0) # T^π(s'|s)

        # A*C_pi(s) = b
        # A = (I - \gamma (T^π)^T)
        # b = P_0
        A = jnp.eye(self.amdp.n_states) - self.amdp.gamma * T_pi.transpose()
        b = self.amdp.p0
        return jnp.linalg.solve(A, b)

    def _solve_amdp(self, mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        return self._functional_solve_amdp(mdp_q_vals, p_pi_of_s_given_o)

    @partial(jit, static_argnums=0)
    def _functional_solve_amdp(self, mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray):

        # Q vals
        amdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

        # V vals
        amdp_v_vals = (amdp_q_vals * self.pi_abs.T).sum(0)

        return {'v': amdp_v_vals, 'q': amdp_q_vals}

    def _create_td_model(self, p_pi_of_s_given_o: jnp.ndarray):
        T_obs_obs, R_obs_obs = self._functional_create_td_model(p_pi_of_s_given_o)
        return MDP(T_obs_obs, R_obs_obs, self.amdp.p0, self.amdp.gamma)

    @partial(jit, static_argnums=0)
    def _functional_create_td_model(self, p_pi_of_s_given_o: jnp.ndarray):
        # this gives us batch x states x 1 and batch x states
        curr_s_given_o = p_pi_of_s_given_o[:, self.obs_idx_product[:, 0]].T[..., None]
        next_o_given_s = jnp.expand_dims(self.amdp.phi[:, self.obs_idx_product[:, 1]].T, 1)

        # outer product here
        o_to_next_o = jnp.expand_dims(curr_s_given_o * next_o_given_s, 1)

        T_contributions = self.amdp.T * o_to_next_o
        T_obs_obs_flat = T_contributions.sum(-1).sum(-1).T
        T_obs_obs = T_obs_obs_flat.reshape(self.amdp.T.shape[0], self.amdp.n_obs, self.amdp.n_obs)

        denom = T_obs_obs_flat.T[..., None, None]
        denom_no_zero = denom + (denom == 0).astype(denom.dtype)

        R_contributions = (self.amdp.R * T_contributions) / denom_no_zero
        R_obs_obs_flat = R_contributions.sum(-1).sum(-1).T
        R_obs_obs = R_obs_obs_flat.reshape(self.amdp.R.shape[0], self.amdp.n_obs, self.amdp.n_obs)

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
