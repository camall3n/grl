import numpy as np

from grl import PolicyEval, load_spec, MDP, AbstractMDP

def check_state(oa_state_vals: dict, c_state_vals: dict):
    oa_v, oa_q = oa_state_vals['v'], oa_state_vals['q']
    c_v, c_q = c_state_vals['v'], c_state_vals['q']
    n_actions = c_q.shape[0]

    def check_indv_vals(oa_vals: np.ndarray, c_vals: np.ndarray):
        left_c_val = c_vals[1]
        right_c_val = c_vals[2]
        assert np.allclose(left_c_val, oa_vals[:n_actions])
        assert np.allclose(right_c_val, oa_vals[n_actions:2 * n_actions])

    check_indv_vals(oa_v, c_v)
    for oa_a_q, c_a_q in zip(oa_q, c_q):
        check_indv_vals(oa_a_q, c_a_q)

def check_obs(oa_obs_vals: dict, c_obs_vals: dict):
    pass

def test_tiger():
    obs_action_spec = load_spec('tiger')
    converted_spec = load_spec('tiger-alt')

    obs_action_mdp = MDP(obs_action_spec['T'], obs_action_spec['R'], obs_action_spec['p0'], obs_action_spec['gamma'])
    obs_action_amdp = AbstractMDP(obs_action_mdp, obs_action_spec['phi'])
    oa_pe = PolicyEval(obs_action_amdp)

    converted_mdp = MDP(converted_spec['T'], converted_spec['R'], converted_spec['p0'], converted_spec['gamma'])
    converted_amdp = AbstractMDP(converted_mdp, converted_spec['phi'])
    c_pe = PolicyEval(converted_amdp)

    for oa_pi, c_pi in zip(obs_action_spec['Pi_phi'], converted_spec['Pi_phi']):
        oa_state_vals, oa_mc_vals, oa_td_vals = oa_pe.run(oa_pi)
        c_state_vals, c_mc_vals, c_td_vals = c_pe.run(c_pi)

        check_state(oa_state_vals, c_state_vals)
        check_obs(oa_mc_vals, c_mc_vals)
        check_obs(oa_td_vals, c_td_vals)

        assert True

if __name__ == "__main__":
    test_tiger()

