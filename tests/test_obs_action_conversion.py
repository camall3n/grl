import numpy as np

from grl import load_spec, MDP, AbstractMDP
from grl.utils.policy_eval import analytical_pe

def check_vals(oa_vals: dict, c_vals: dict):
    oa_v, oa_q = oa_vals['v'], oa_vals['q']
    c_v, c_q = c_vals['v'], c_vals['q']

    assert np.allclose(oa_v, c_v)
    assert np.allclose(oa_q, c_q)

def test_tiger():
    obs_action_spec = load_spec('tiger')
    converted_spec = load_spec('tiger-alt-act')

    obs_action_mdp = MDP(obs_action_spec['T'], obs_action_spec['R'], obs_action_spec['p0'],
                         obs_action_spec['gamma'])
    obs_action_amdp = AbstractMDP(obs_action_mdp, obs_action_spec['phi'])

    converted_mdp = MDP(converted_spec['T'], converted_spec['R'], converted_spec['p0'],
                        converted_spec['gamma'])
    converted_amdp = AbstractMDP(converted_mdp, converted_spec['phi'])

    for oa_pi, c_pi in zip(obs_action_spec['Pi_phi'], converted_spec['Pi_phi']):
        oa_state_vals, oa_mc_vals, oa_td_vals, _ = analytical_pe(oa_pi, obs_action_amdp)
        c_state_vals, c_mc_vals, c_td_vals, _ = analytical_pe(c_pi, converted_amdp)

        check_vals(oa_state_vals, c_state_vals)
        check_vals(oa_mc_vals, c_mc_vals)
        check_vals(oa_td_vals, c_td_vals)

    print("All tests passed.")

if __name__ == "__main__":
    test_tiger()
