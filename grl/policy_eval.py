import numpy as np

from mdp import MDP

class PolicyEval:
    def __init__(self, amdp, pi):
        """
        :param amdp:   AMDP
        :param pi:     list/array of actions
        """
        self.amdp = amdp
        self.pi = pi

    def run(self):
        # MC*
        mdp_vals = self.solve_mdp(self.amdp)
        weights = self.get_weights()
        amdp_vals = self.solve_amdp(mdp_vals, weights)

        # TD
        td_vals = self.solve_mdp(self.create_td_model(weights))

        return mdp_vals, amdp_vals, td_vals

    def solve_mdp(self, mdp):
        """
        Solves for V using linear equations
        For all s, V_pi(s) = sum_(s')[T(s,pi(s),s') * (R(s,pi(s),s') + gamma*V_pi(s'))]
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

    def get_weights(self):
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
                t =  self.amdp.T[self.pi[prev_s],prev_s,s]
                # t = self.amdp.gamma * self.amdp.T[self.pi[prev_s],prev_s,s]
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
            col = self.amdp.phi[:,:,i].copy().astype('float')[self.pi[i]]
            col *= weights
            col /= col.sum()
            v = mdp_vals * col
            amdp_vals[i] += v.sum()

        return amdp_vals

    def create_td_model(self, weights):
        T_obs_state = np.zeros((len(self.amdp.T), self.amdp.n_obs, self.amdp.n_states))
        T_state_obs =  np.zeros((len(self.amdp.T), self.amdp.n_states, self.amdp.n_obs))

        for i in range(self.amdp.n_obs):
            col = self.amdp.phi[:,:,i].copy().astype('float')[self.pi[i]] #TODO: phi indexing correct?
            w = col.copy() * weights

            w /= w.sum()
            T = self.amdp.T * w[:,None]
            T_obs_state[:,i] = T.sum(1)
            
            T = self.amdp.T * weights[:,None]
            T = self.amdp.T * col
            T_state_obs[:,:,i] = T.sum(2)

        R_obs_obs = np.zeros((len(self.amdp.R), self.amdp.n_obs, self.amdp.n_obs))
        for i in range(self.amdp.n_obs):
            col = self.amdp.phi[:,:,i].copy().astype('float')[self.pi[i]]
            w = weights * col.copy() # Prob of being in each state * prob of it emitting curr obs i
            w = w[:,None] * self.amdp.T

            for j in range(self.amdp.n_obs):
                w2 = w.copy()
                col2 = self.amdp.phi[:,:,j].copy().astype('float')[self.pi[i]]
                w2 *= col2 # Prob of next_state emitting next_obs j

                sum = w2.reshape((len(self.amdp.R),-1)).sum((1))
                if np.all(sum != 0):
                    w2 /= sum[:,None,None]
                R = (self.amdp.R * w2)
                R_obs_obs[:,i,j] = R.reshape((len(self.amdp.R),-1)).sum(1)

        # Convert from obs->state to obs->obs
        T_obs_obs = np.zeros((len(self.amdp.T), self.amdp.n_obs, self.amdp.n_obs))
        for i in range(self.amdp.n_obs):
            col = self.amdp.phi[:,:,i].copy().astype('float')[self.pi[i]]
            T_obs_obs[:,:,i] = (T_obs_state*(col>0)).sum(2)

        return MDP(T_obs_obs, R_obs_obs, self.amdp.gamma)
