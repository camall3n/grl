import numpy as np

from grl.grl import PolicyEval, load_spec, MDP, AbstractMDP

def assert_pe_results(spec, answers):
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])

    for i, pi in enumerate(spec['Pi_phi']):
        pe = PolicyEval(amdp)
        results = pe.run(pi, True)

        for k in answers.keys():
            for j, res in enumerate(results):
                assert (np.allclose(answers[k][i][j], res[k]))

def test_example_3():
    spec = load_spec('example_3')
    answers = {
        'v': [[
            np.array([3.8125, 4.5, 0.75, 0]), # mdp
            np.array([3.8125, 3.5625, 0]), # amdp / mc*
            np.array([3.8125, 3.5625, 0]), # td
        ]],
        'q': [[
            np.array([
                [4.5, 1.75], # mdp
                [5, 3],
                [0, 3],
                [0, 0]
            ]),
            np.array([
                [4.5, 1.75], # amdp
                [3.75, 3],
                [0, 0]
            ]),
            np.array([
                [3.5625, 4.5625], # td
                [3.75, 3],
                [0, 0]
            ])
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_11():
    spec = load_spec('example_11')
    answers = {
        'v': [[
            np.array([1 / 7, 2 / 7, 4 / 7, 0]),
            np.array([1 / 7, 3 / 7, 0]),
            np.array([1 / 5, 2 / 5, 0]),
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_13():
    spec = load_spec('example_13')
    answers = {
        'v': [[
            np.array([1 / 7, 2 / 7, 4 / 7, 0]),
            np.array([3 / 7, 2 / 7, 0]),
            np.array([2 / 5, 1 / 5, 0]),
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_14():
    spec = load_spec('example_14')
    answers = {
        'v': [[np.array([1, 0.5, 1, 0]),
               np.array([0.875, 1, 0]),
               np.array([0.875, 1, 0])],
              [np.array([0.5, 1, 1, 0]),
               np.array([0.625, 1, 0]),
               np.array([0.625, 1, 0])],
              [np.array([0.75, 0.75, 1, 0]),
               np.array([0.75, 1, 0]),
               np.array([0.75, 1, 0])]],
        'q': [[
            np.array([[1., 0.5], [0.5, 1.], [1., 1.], [0., 0.]]),
            np.array([[0.875, 0.625], [1., 1.], [0., 0.]]),
            np.array([[0.875, 0.625], [1., 1.], [0., 0.]]),
        ],
              [
                  np.array([[1., 0.5], [0.5, 1.], [1., 1.], [0., 0.]]),
                  np.array([[0.875, 0.625], [1., 1.], [0., 0.]]),
                  np.array([[0.875, 0.625], [1., 1.], [0., 0.]]),
              ],
              [
                  np.array([[1., 0.5], [0.5, 1.], [1., 1.], [0., 0.]]),
                  np.array([[0.875, 0.625], [1., 1.], [0., 0.]]),
                  np.array([[0.875, 0.625], [1., 1.], [0., 0.]]),
              ]]
    }

    assert_pe_results(spec, answers)
