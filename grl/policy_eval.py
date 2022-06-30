import logging

import jax.numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from .mdp import MDP

class PolicyEval:
    def __init__(self, amdp, no_gamma, verbose=True):
        """
        :param amdp:    AMDP
        :param verbose: log everything
        """
        self.amdp = amdp
        self.no_gamma = no_gamma
        self.verbose = verbose

    def run(self, pi_abs):
        """
        :param pi_abs:   policy to evaluate, defined over abstract state space
        """
        self.pi_abs = pi_abs
        self.pi_ground = self.amdp.get_ground_policy(pi_abs)

        # MC*
        mdp_vals = self._solve_mdp(self.amdp, self.pi_ground)
        occupancy = self._get_occupancy()
        amdp_vals = self._solve_amdp(mdp_vals['q'], occupancy)

        if self.verbose:
            logging.info(f'occupancy:\n {occupancy}')

        # TD
        td_vals = self._solve_mdp(self._create_td_model(occupancy), self.pi_abs)

        return mdp_vals, amdp_vals, td_vals

    def _solve_mdp(self, mdp, pi):
        """
        Solves for V using linear equations.
        For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
        """
        Pi_pi = pi.transpose()[..., None]
        T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)
        R_pi = (Pi_pi * mdp.T * mdp.R).sum(axis=0).sum(axis=-1) # R^π(s)

        # A*V_pi(s) = b
        # A = (I - \gamma (T^π))
        # b = R^π
        A = (np.eye(mdp.n_states) - mdp.gamma * T_pi)
        b = R_pi
        v_vals = np.linalg.solve(A, b)

        R_sa = (mdp.T * mdp.R).sum(axis=-1) # R(s,a)
        q_vals = (R_sa + (mdp.gamma * mdp.T @ v_vals))

        return {'v': v_vals, 'q': q_vals}

    def _get_occupancy(self):
        """
        Finds the visitation count, C_pi(s), of each state.
        For all s, C_pi(s) = p0(s) + sum_s^ sum_a[C_pi(s^) * gamma * T(s|a,s^) * pi(a|s^)],
          where s^ is the prev state
        """
        Pi_pi = self.pi_ground.transpose()[..., None]
        T_pi = (Pi_pi * self.amdp.T).sum(axis=0) # T^π(s'|s)

        # A*C_pi(s) = b
        # A = (I - \gamma (T^π)^T)
        # b = P_0
        gamma = 1 if self.no_gamma else self.amdp.gamma
        A = np.eye(self.amdp.n_states) - gamma * T_pi.transpose()
        b = self.amdp.p0
        return np.linalg.solve(A, b)

    def _solve_amdp(self, mdp_q_vals, occupancy):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        amdp_q_vals = np.zeros((self.amdp.n_actions, self.amdp.n_obs))

        # Q vals
        for ob in range(self.amdp.n_obs):
            col = self.amdp.phi[:, ob].copy().astype('float')
            col *= occupancy
            col /= col.sum()
            weighted_q = (mdp_q_vals * col).sum(1)
            amdp_q_vals = amdp_q_vals.at[:, ob].set(weighted_q)

        # V vals
        amdp_v_vals = (amdp_q_vals * self.pi_abs.T).sum(0)

        return {'v': amdp_v_vals, 'q': amdp_q_vals}

    def _create_td_model(self, occupancy):
        """
        Generates effective TD(0) model
        """
        T_obs_obs = np.zeros((len(self.amdp.T), self.amdp.n_obs, self.amdp.n_obs))
        R_obs_obs = np.zeros((len(self.amdp.R), self.amdp.n_obs, self.amdp.n_obs))
        for curr_ob in range(self.amdp.n_obs):
            # phi is |S|x|O|
            ###### curr_a = self.pi[curr_ob]
            # compute p_π(o|s) for all s
            p_π_of_o_given_s = self.amdp.phi[:, curr_ob].copy().astype('float')
            # want p_π(s|o) ∝ p_π(o|s)p(s) = p_π_of_o_given_s * occupancy
            w = occupancy * p_π_of_o_given_s # Count of being in each state * prob of it emitting curr_ob
            p_π_of_s_given_o = (w / w.sum())[:, None]

            for next_ob in range(self.amdp.n_obs):
                # Q: what action should this be? [self.pi[i]]
                p_π_of_op_given_sp = self.amdp.phi[:, next_ob].copy().astype('float')

                # T
                T_contributions = (self.amdp.T * p_π_of_s_given_o * p_π_of_op_given_sp)
                # sum over s', then over s
                # T_obs_obs[:,curr_ob,next_ob] = T_contributions.sum(2).sum(1)
                T_obs_obs = T_obs_obs.at[:, curr_ob, next_ob].set(T_contributions.sum(2).sum(1))

                # R
                R_contributions = self.amdp.R * T_contributions
                denom = T_obs_obs[:, curr_ob, next_ob][:, None, None]
                denom = np.where(denom == 0, 1,
                                 denom) # Avoid divide by zero (there may be a better way)
                R_contributions /= denom

                # R_obs_obs[:,curr_ob,next_ob] = R_contributions.sum(2).sum(1)
                R_obs_obs = R_obs_obs.at[:, curr_ob, next_ob].set(R_contributions.sum(2).sum(1))

        if self.verbose:
            logging.info(f'T_bar:\n {T_obs_obs}')
            logging.info(f'R_bar:\n {R_obs_obs}')

        return MDP(T_obs_obs, R_obs_obs, self.amdp.gamma)

    ##########
    # Helpers for gradient/heatmap stuff
    ##########

    def mse_loss_v(self, pi):
        """
        sum_o [V_td^pi(o) - V_mc^pi(o)]^2 
        """
        _, amdp_vals, td_vals = self.run(pi)
        diff = amdp_vals['v'] - td_vals['v']
        return (diff**2).mean()

    def mse_loss_q(self, pi):
        """
        sum_o sum_a [Q_td^pi(o,a) - Q_mc^pi(o,a)]^2 
        """
        _, amdp_vals, td_vals = self.run(pi)
        diff = amdp_vals['q'] - td_vals['q']
        return (diff**2).mean()

    def max_loss_v(self, pi):
        """
        max_o abs[V_td^pi(o) - V_mc^pi(o)]
        """
        _, amdp_vals, td_vals = self.run(pi)
        return np.abs(amdp_vals['v'] - td_vals['v']).max()

    def max_loss_q(self, pi):
        """
        max_o max_a abs[Q_td^pi(o,a) - Q_mc^pi(o,a)]
        """
        _, amdp_vals, td_vals = self.run(pi)
        return np.abs(amdp_vals['q'] - td_vals['q']).max()
