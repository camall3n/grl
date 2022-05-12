import numpy as np

def policy_eval(pomdp, policy):
    """
    For all s, V_pi(s) = sum_(s')[T(s,pi(s),s') * (R(s,pi(s),s') + gamma*V_pi(s'))]

    :param pomdp: AMDP
    :param policy: list/array
    """

    # Create list of linear equations
    a = []
    b = []
    for obs in range(pomdp.base_mdp.n_obs):
        a_t = np.zeros(pomdp.base_mdp.n_obs)
        a_t[obs] = -1 # subtract V_pi(obs) to right side
        b_t = 0
        for next_obs in range(pomdp.base_mdp.n_obs):
            t = pomdp.T[policy[obs],obs,next_obs]
            b_t -= t * pomdp.R[policy[obs],obs,next_obs] # subtract constants to left side
            a_t[next_obs] += t * pomdp.gamma
        
        a.append(a_t)
        b.append(b_t)

    # Solve mdp
    mdp_vals = np.linalg.solve(a, b)

    # Solve amdp 
    amdp_vals = np.zeros(pomdp.n_obs)
    for i in range(pomdp.phi.shape[-1]):
        col = pomdp.phi[:,:,i].copy().astype('float')[0]
        col /= col.sum()
        v = mdp_vals * col
        amdp_vals[i] += v.sum()

    return mdp_vals, amdp_vals
