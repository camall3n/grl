import numpy as np

from grl.grl import PolicyEval, load_spec, MDP, AbstractMDP

def assert_pe_results(spec, answers):
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])

    for i, pi in enumerate(spec['Pi_phi']):
        pe = PolicyEval(amdp, pi)
        res = pe.run(True)

        for j, res in enumerate(res):
            assert(np.allclose(answers[i][j], res))

def test_example_3():
    spec = load_spec('example_3')
    answers = [[
        np.array([3.8125, 4.5, 0.75, 0, 0, 0, 0]),
        np.array([3.8125, 3.5625, 0]),
        np.array([3.8125, 3.5625, 0]),
    ]]

    assert_pe_results(spec, answers)

def test_example_11():
    spec = load_spec('example_11')
    answers = [[
        np.array([0.14285714, 0.28571429, 0.57142857, 0]),
        np.array([0.14285714, 0.42857143, 0]),
        np.array([0.2, 0.4, 0]),
    ]]

    assert_pe_results(spec, answers)

def test_example_13():
    spec = load_spec('example_13')
    answers = [[
        np.array([0.14285714, 0.28571429, 0.57142857, 0]),
        np.array([0.42857143, 0.28571429, 0]),
        np.array([0.4, 0.2, 0]),
    ]]

    assert_pe_results(spec, answers)

def test_example_14():
    spec = load_spec('example_14')
    answers = [[
        np.array([1, 0.5, 1, 0]),
        np.array([0.875, 1, 0]),
        np.array([0.875, 1, 0])
    ],
    [
        np.array([0.5, 1, 1, 0]),
        np.array([0.625, 1, 0]),
        np.array([0.625, 1, 0])
    ],
    [
        np.array([0.75, 0.75, 1, 0]),
        np.array([0.75, 1, 0]),
        np.array([0.75, 1, 0])
    ]]

    assert_pe_results(spec, answers)
