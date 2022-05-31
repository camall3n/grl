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
            for next_s in range(mdp.n_states):
                t = mdp.T * self.pi[s][:, None, None]
                t = t.sum(0)[s,next_s]
                r = mdp.R * self.pi[s][:, None, None]
                r = r.sum(0)[s,next_s]
                a_t[next_s] += t * mdp.gamma # add V_pi(s') to right side
                b_t -= t * r # subtract constants to left side

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
                t = self.amdp.T * self.pi[prev_s][:, None, None]
                t = t.sum(0)[prev_s,s]
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
            col = self.amdp.phi[:,:,i] * self.pi[i][:, None]
            col = col * weights
            with np.errstate(invalid='ignore'):
                col /= col.sum()
            col = np.nan_to_num(col)
            v = mdp_vals * col
            amdp_vals[i] += v.sum()

        return amdp_vals

    def create_td_model(self, weights):
        """
        Generates effective TD(0) model
        """
        T_obs_obs = np.zeros((len(self.amdp.T), self.amdp.n_obs, self.amdp.n_obs))
        R_obs_obs = np.zeros((len(self.amdp.R), self.amdp.n_obs, self.amdp.n_obs))
        for i in range(self.amdp.n_obs):
            col = self.amdp.phi[:,:,i] * self.pi[i][:, None]
            col = col.sum(0).squeeze()
            w = weights * col # Prob of being in each state * prob of it emitting curr obs i
            with np.errstate(invalid='ignore'):
                w_t = (w / w.sum())[:,None] * self.amdp.T
            w_t = np.nan_to_num(w_t)
            w_r = w[:,None] * self.amdp.T

            for j in range(self.amdp.n_obs):
                col2 = self.amdp.phi[:,:,j] * self.pi[i][:, None]
                col2 = col2.sum(0).squeeze()

                # T
                w2 = w_t * col2 # Further weight by prob of next_state emitting next_obs j
                T_obs_obs[:,i,j] = w2.reshape((len(self.amdp.T),-1)).sum(1)

                # R
                w2 = w_r * col2
                with np.errstate(invalid='ignore'):
                    w2 /= w2.reshape(len(self.amdp.R),-1).sum(1)[:,None,None] # Normalize probs within each action
                w2 = np.nan_to_num(w2)
                R = self.amdp.R * w2
                R_obs_obs[:,i,j] = R.reshape(len(self.amdp.R),-1).sum(1)
            
        return MDP(T_obs_obs, R_obs_obs, self.amdp.gamma)
