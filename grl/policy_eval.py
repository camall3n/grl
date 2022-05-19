import numpy as np

def policy_eval(pomdp, pi):
    """
    :param pomdp:   AMDP
    :param pi:      list/array of actions
    """

    # Create list of linear equations
    # For all s, V_pi(s) = sum_(s')[T(s,pi(s),s') * (R(s,pi(s),s') + gamma*V_pi(s'))]
    a = []
    b = []
    for s in range(pomdp.n_states):
        a_t = np.zeros(pomdp.n_states)
        a_t[s] = -1 # subtract V_pi(s) to right side
        b_t = 0
        possible_next_ss = np.where(pomdp.T[pi[s],s] != 0.)[0]
        for next_s in possible_next_ss:
            t = pomdp.T[pi[s],s,next_s]
            b_t -= t * pomdp.R[pi[s],s,next_s] # subtract constants to left side
            a_t[next_s] += t * pomdp.gamma

        a.append(a_t)
        b.append(b_t)

    # Solve mdp
    mdp_vals = np.linalg.solve(a, b)

    # Solve amdp 
    # Find the likelihood, P_pi(s), of reaching each state
    #   and use this likelihood to weight how much of the sum it gets
    # For all s, P_pi(s) = p0(s) + sum_s"[P_pi(s") * gamma * T(s",pi(s"),s)],
    #   where s" is the prev state
    a = []
    for s in range(pomdp.n_states):
        a_t = np.zeros(pomdp.n_states)
        a_t[s] = -1 # subtract P_pi(s) to right side
        for prev_s in range(pomdp.n_states):
            t = pomdp.gamma * pomdp.T[pi[prev_s],prev_s,s]
            a_t[prev_s] += t

        a.append(a_t)
        
    b = -1 * pomdp.p0 # subtract p0(s) to left side
    weights = np.linalg.solve(a, b)

    amdp_vals = np.zeros(pomdp.n_obs)
    for i in range(pomdp.phi.shape[-1]):
        col = pomdp.phi[:,:,i].copy().astype('float')[pi[i]]
        col *= weights
        col /= col.sum()
        v = mdp_vals * col
        amdp_vals[i] += v.sum()

    return mdp_vals, amdp_vals
