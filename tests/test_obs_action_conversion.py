
from grl import PolicyEval, load_spec, MDP, AbstractMDP

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
        oa_results = oa_pe.run(oa_pi)
        c_results = c_pe.run(c_pi)

        assert True

if __name__ == "__main__":
    test_tiger()

