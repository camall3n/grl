import logging

import jax.numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from .mdp import MDP

class PolicyEval:

    def __init__(self, amdp, verbose=True):
        """
        :param amdp:    AMDP
        :param verbose: log everything
        """
        self.amdp = amdp
        self.verbose = verbose

    def run(self, pi_abs, no_gamma):
        """
        :param pi_abs:   policy to evaluate, defined over abstract state space
        :param no_gamma: if True, do not discount the occupancy expectation
        """
        self.pi_abs = pi_abs
        self.pi_ground = self.amdp.get_ground_policy(pi_abs)

        mdp_vals = {}
        amdp_vals = {}
        td_vals = {}

        # MC*
        mdp_vals = self._solve_mdp(self.amdp, self.pi_ground)
        occupancy = self._get_occupancy(no_gamma)
        amdp_vals = self._solve_amdp(mdp_vals['q'], occupancy)

        if self.verbose:
            logging.info(f'occupancy:\n {occupancy}')

        # TD
        td_vals = self._solve_mdp(self._create_td_model(occupancy), self.pi_abs)

        return mdp_vals, amdp_vals, td_vals

    def _solve_mdp(self, mdp, pi):
        """
        Solves for V using linear equations.
        For all s, V_pi(s) = sum_s'[T(s,pi(s),s') * (R(s,pi(s),s') + gamma * V_pi(s'))]
        """
        Pi_pi = pi.transpose()[...,None]
        T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)
        R_pi = (Pi_pi * mdp.T * mdp.R).sum(axis=0).sum(axis=-1) # R^π(s)

        # A*V_pi(s) = b
        A = (np.eye(mdp.n_states) - mdp.gamma * T_pi)
        b = R_pi
        v_vals = np.linalg.solve(A, b)

        # Q vals
        q_vals = np.zeros((mdp.n_states, mdp.n_actions))
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                for sp in range(mdp.n_states):
                    # q_vals[s,a] += mdp.R[a,s,sp] + mdp.gamma * mdp.T[a,s,sp] * v_vals[sp]
                    q_vals = q_vals.at[s, a].set(q_vals[s, a] + mdp.T[a, s, sp] *
                                                 (mdp.R[a, s, sp] + mdp.gamma * v_vals[sp]))

        return {'v': v_vals, 'q': q_vals}

    def _get_occupancy(self, no_gamma):
        """
        Finds the visitation count, C_pi(s), of each state.
        For all s, C_pi(s) = p0(s) + sum_s^[C_pi(s^) * gamma * T(s^,pi(s^),s)],
          where s^ is the prev state
        """
        # Each index of this list corresponds to one linear equation
        # b = A*C_pi(s)
        A = []
        for s in range(self.amdp.n_states):
            a_t = np.zeros(self.amdp.n_states)
            # a_t[s] = -1 # subtract C_pi(s) to right side
            a_t = a_t.at[s].set(-1)
            for prev_s in range(self.amdp.n_states):
                T_pi = np.tensordot(self.pi_ground[prev_s], self.amdp.T, axes=1)
                t = T_pi[prev_s, s]
                if not no_gamma:
                    t *= self.amdp.gamma
                # a_t[prev_s] += t
                a_t = a_t.at[prev_s].set(a_t[prev_s] + t)

            A.append(a_t)

        b = -1 * self.amdp.p0 # subtract p0(s) to left side
        return np.linalg.solve(A, b)

    def _solve_amdp(self, mdp_q_vals, occupancy):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        amdp_q_vals = np.zeros((self.amdp.n_obs, self.amdp.n_actions))

        # Q vals
        for ob in range(self.amdp.n_obs):
            col = self.amdp.phi[:, ob].copy().astype('float')
            col *= occupancy
            col /= col.sum()
            weighted_q = (mdp_q_vals * col[:, None]).sum(0)
            amdp_q_vals = amdp_q_vals.at[ob].set(weighted_q)

        # V vals
        amdp_v_vals = (amdp_q_vals * self.pi_abs).sum(1)

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
            w = occupancy * p_π_of_o_given_s # Prob of being in each state * prob of it emitting curr obs i
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
