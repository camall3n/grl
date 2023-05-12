import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import environment, MDP, AbstractMDP
from grl.utils.mdp import get_p_s_given_o, functional_create_td_model, amdp_get_occupancy
from grl.utils.policy_eval import functional_solve_amdp, functional_solve_mdp

# Original, serial functions
def solve_amdp(amdp, mdp_q_vals, pi_abs, occupancy):
    """
    Weights the value contribution of each state to each observation for the amdp
    """
    amdp_q_vals = jnp.zeros((amdp.n_actions, amdp.n_obs))

    # Q vals
    for ob in range(amdp.n_obs):
        p_of_o_given_s = amdp.phi[:, ob].copy().astype('float')
        w = occupancy * p_of_o_given_s
        # Skip this ob (leave vals at 0) if w is full of 0s
        # as this means it will never be occupied
        # and normalizing comes up as nans
        if np.all(w == 0):
            continue
        p_π_of_s_given_o = w / w.sum()
        weighted_q = (mdp_q_vals * p_π_of_s_given_o).sum(1)
        amdp_q_vals = amdp_q_vals.at[:, ob].set(weighted_q)

    # V vals
    amdp_v_vals = (amdp_q_vals * pi_abs.T).sum(0)

    return {'v': amdp_v_vals, 'q': amdp_q_vals}

def create_td_model(amdp, occupancy):
    """
    Generates effective TD(0) model
    """
    T_obs_obs = jnp.zeros((len(amdp.T), amdp.n_obs, amdp.n_obs))
    R_obs_obs = jnp.zeros((len(amdp.R), amdp.n_obs, amdp.n_obs))
    for curr_ob in range(amdp.n_obs):
        # phi is |S|x|O|
        ###### curr_a = self.pi[curr_ob]
        # compute p_π(o|s) for all s
        p_of_o_given_s = amdp.phi[:, curr_ob].copy().astype('float')
        # want p_π(s|o) ∝ p_π(o|s)p(s) = p_π_of_o_given_s * occupancy
        w = occupancy * p_of_o_given_s # Count of being in each state * prob of it emitting curr_ob
        # Skip this ob (leave vals at 0) if w is full of 0s
        # as this means it will never be occupied
        # and normalizing comes up as nans
        if jnp.all(w == 0):
            continue
        p_π_of_s_given_o = (w / w.sum())[:, None]

        for next_ob in range(amdp.n_obs):
            # Q: what action should this be? [self.pi[i]]
            p_π_of_op_given_sp = amdp.phi[:, next_ob].copy().astype('float')

            # T
            T_contributions = (amdp.T * p_π_of_s_given_o * p_π_of_op_given_sp)
            # sum over s', then over s
            # T_obs_obs[:,curr_ob,next_ob] = T_contributions.sum(2).sum(1)
            T_obs_obs = T_obs_obs.at[:, curr_ob, next_ob].set(T_contributions.sum(2).sum(1))

            # R
            R_contributions = amdp.R * T_contributions
            denom = T_obs_obs[:, curr_ob, next_ob][:, None, None]
            denom = np.where(denom == 0, 1,
                             denom) # Avoid divide by zero (there may be a better way)
            R_contributions /= denom

            # R_obs_obs[:,curr_ob,next_ob] = R_contributions.sum(2).sum(1)
            R_obs_obs = R_obs_obs.at[:, curr_ob, next_ob].set(R_contributions.sum(2).sum(1))

    return MDP(T_obs_obs, R_obs_obs, amdp.p0, amdp.gamma)

# Given an environment specification, compare serial functions we were using (defined above)
# and functional jax-based functions defined now in utils/policy_eval.
def indv_spec_jaxify_pe_funcs(spec):
    pi = spec['Pi_phi'][0]
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    amdp = AbstractMDP(mdp, spec['phi'])
    # MC*
    pi_ground = amdp.get_ground_policy(pi)
    mdp_v, mdp_q = functional_solve_mdp(pi_ground, mdp)
    occupancy = amdp_get_occupancy(pi, amdp)
    p_pi_of_s_given_o = get_p_s_given_o(amdp.phi, occupancy)
    func_mc_vals = functional_solve_amdp(mdp_q, p_pi_of_s_given_o, pi)

    mc_vals = solve_amdp(amdp, mdp_q, pi, occupancy)

    assert np.all(np.isclose(mc_vals['v'], func_mc_vals['v'])) and np.all(
        np.isclose(mc_vals['q'], func_mc_vals['q']))

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, amdp)

    func_td_mdp = MDP(T_obs_obs, R_obs_obs, amdp.p0, amdp.gamma)

    td_mdp = create_td_model(amdp, occupancy)

    assert np.all(np.isclose(func_td_mdp.T, td_mdp.T))
    assert np.all(np.isclose(func_td_mdp.R, td_mdp.R))

# Test on two example environments!
def test_jaxify_pe_funcs():
    spec_strings = ['example_3', 'tmaze_5_two_thirds_up']
    for spec_str in spec_strings:
        spec = environment.load_spec(spec_str, memory_id=None)
        indv_spec_jaxify_pe_funcs(spec)

if __name__ == "__main__":
    test_jaxify_pe_funcs()
