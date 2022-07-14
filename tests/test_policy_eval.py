import numpy as np

from grl.grl import PolicyEval, load_spec, MDP, AbstractMDP, memory_cross_product

def assert_pe_results(spec, answers, use_memory=False):
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])
    policies = spec['Pi_phi']

    if use_memory:
        amdp = memory_cross_product(amdp, spec['T_mem'])
        policies = spec['Pi_phi_x']

    for i, pi in enumerate(policies):
        pe = PolicyEval(amdp, no_gamma=True)
        results = pe.run(pi)

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
                [4.5, 5, 0, 0], # mdp
                [1.75, 3, 3, 0]
            ]),
            np.array([
                [4.5, 3.75, 0], # amdp
                [1.75, 3, 0],
            ]),
            np.array([
                [3.5625, 3.75, 0], # td
                [4.5625, 3, 0],
            ])
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_7_memory():
    spec = load_spec('example_7')
    answers = {
        'v': [[
            np.array([1.25, 0.25, 0.5, 0., 0., 1., 0., 0.]),
            np.array([1.25, 1., 0.5, 0., 0., 0.]),
            np.array([1.25, 1., 0.5, 0., 0., 0.]),
        ]],
        'q': [[
            np.array([[
                [0.25, 0.25, 0.5, 0., 1., 1., 0., 0.],
                [1.25, 1.25, 0.5, 0., 0., 0., 0., 0.],
            ]]),
            np.array([[
                [0.25, 1., 0.5, 0., 0., 0.],
                [1.25, 0., 0.5, 0., 0., 0.],
            ]]),
            np.array([[
                [0.25, 1., 0.5, 0., 0., 0.],
                [1.25, 0., 0.5, 0., 0., 0.],
            ]]),
        ]],
    }

    assert_pe_results(spec, answers, use_memory=True)

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
        'q': [
            [
                np.array([
                    [1, 0.5, 1, 0],
                    [0.5, 1, 1, 0],
                ]),
                np.array([
                    [.875, 1, 0],
                    [.625, 1, 0],
                ]),
                np.array([
                    [.875, 1, 0],
                    [.625, 1, 0],
                ]),
            ],
            [
                np.array([
                    [1, 0.5, 1, 0],
                    [0.5, 1, 1, 0],
                ]),
                np.array([
                    [.875, 1, 0],
                    [.625, 1, 0],
                ]),
                np.array([
                    [.875, 1, 0],
                    [.625, 1, 0],
                ]),
            ],
            [
                np.array([
                    [1, 0.5, 1, 0],
                    [0.5, 1, 1, 0],
                ]),
                np.array([
                    [.875, 1, 0],
                    [.625, 1, 0],
                ]),
                np.array([
                    [.875, 1, 0],
                    [.625, 1, 0],
                ]),
            ],
        ]
    }

    assert_pe_results(spec, answers)
