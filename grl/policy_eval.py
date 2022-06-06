import logging

import numpy as np

from .mdp import MDP

class PolicyEval:
    def __init__(self, amdp, pi):
        """
        :param amdp:   AMDP
        :param pi:     A policy
        """
        self.amdp = amdp
        self.pi = pi

    def run(self, no_gamma):
        """
        :param no_gamma: if True, do not discount the weighted average value expectation
        """
        # MC*
        mdp_vals = self.solve_mdp(self.amdp)
        weights = self.get_weights(no_gamma)
        amdp_vals = self.solve_amdp(mdp_vals, weights)

        # TD
        td_vals = self.solve_mdp(self.create_td_model(weights))

        return mdp_vals, amdp_vals, td_vals

    def solve_mdp(self, mdp):
        """
        Solves for V using linear equations.
        For all s, V_pi(s) = sum_s'[T(s,pi(s),s') * (R(s,pi(s),s') + gamma * V_pi(s'))]
        """
        a = []
        b = []
        for s in range(mdp.n_states):
            a_t = np.zeros(mdp.n_states)
            a_t[s] = -1 # subtract V_pi(s) to right side
            b_t = 0
            possible_next_ss = np.where(mdp.T[self.pi[s],s] != 0.)[0]
            for next_s in possible_next_ss:
                t = mdp.T[self.pi[s],s,next_s]
                b_t -= t * mdp.R[self.pi[s],s,next_s] # subtract constants to left side
                a_t[next_s] += t * mdp.gamma

            a.append(a_t)
            b.append(b_t)

        return np.linalg.solve(a, b)

    def get_weights(self, no_gamma):
        """
        Finds the likelihood, P_pi(s), of reaching each state.
        For all s, P_pi(s) = p0(s) + sum_s"[P_pi(s") * gamma * T(s",pi(s"),s)],
          where s" is the prev state
        """
        a = []
        for s in range(self.amdp.n_states):
            a_t = np.zeros(self.amdp.n_states)
            a_t[s] = -1 # subtract P_pi(s) to right side
            for prev_s in range(self.amdp.n_states):
                t = self.amdp.T[self.pi[prev_s],prev_s,s]
                if not no_gamma:
                    t *= self.amdp.gamma
                a_t[prev_s] += t

            a.append(a_t)

        b = -1 * self.amdp.p0 # subtract p0(s) to left side
        return np.linalg.solve(a, b)

    def solve_amdp(self, mdp_vals, weights):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        amdp_vals = np.zeros(self.amdp.n_obs)
        for i in range(self.amdp.n_obs):
            col = self.amdp.phi[:,i].copy().astype('float')#[self.pi[i]]
            col *= weights
            col /= col.sum()
            v = mdp_vals * col
            amdp_vals[i] += v.sum()

        return amdp_vals

    def create_td_model(self, occupancy):
        """
        Generates effective TD(0) model
        """
        logging.info(f'occupancy: {occupancy}')
        T_obs_obs = np.zeros((len(self.amdp.T), self.amdp.n_obs, self.amdp.n_obs))
        R_obs_obs = np.zeros((len(self.amdp.R), self.amdp.n_obs, self.amdp.n_obs))
        for curr_ob in range(self.amdp.n_obs):
            # phi is |S|x|O|
            ###### curr_a = self.pi[curr_ob]
            # compute p_π(o|s) for all s
            p_π_of_o_given_s = self.amdp.phi[:, curr_ob].copy().astype('float')
            # want p_π(s|o) ∝ p_π(o|s)p(s) = p_π_of_o_given_s * occupancy
            w = occupancy * p_π_of_o_given_s # Prob of being in each state * prob of it emitting curr obs i
            p_π_of_s_given_o = (w / w.sum())[:,None]

            for next_ob in range(self.amdp.n_obs):
                # Q: what action should this be? [self.pi[i]]
                p_π_of_op_given_sp = self.amdp.phi[:,next_ob].copy().astype('float')

                # T
                T_contributions = (self.amdp.T * p_π_of_s_given_o * p_π_of_op_given_sp)
                # sum over s', then over s
                T_obs_obs[:,curr_ob,next_ob] = T_contributions.sum(2).sum(1)

                # R
                with np.errstate(invalid='ignore'):
                    R_contributions = np.nan_to_num(self.amdp.R * T_contributions / T_obs_obs[:,curr_ob,next_ob][:, None, None])
                R_obs_obs[:,curr_ob,next_ob] = R_contributions.sum(2).sum(1)


        logging.info(f'T_bar: {T_obs_obs}')
        logging.info(f'R_bar: {R_obs_obs}')
        return MDP(T_obs_obs, R_obs_obs, self.amdp.gamma)
