import numpy as np
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import load_spec, MDP, POMDP, memory_cross_product
from grl.utils.policy_eval import analytical_pe
from grl.memory import get_memory

def assert_pe_results(spec, answers, mem_params=None):
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])
    policies = spec['Pi_phi']

    if mem_params is not None:
        amdp = memory_cross_product(mem_params, amdp)
        policies = spec['Pi_phi_x']

    for i, pi in enumerate(policies):
        results = analytical_pe(pi, amdp)

        for k in answers.keys():
            for j, res in enumerate(results):
                if k in res:
                    assert (np.allclose(answers[k][i][j], res[k]))

def test_example_3():
    spec = load_spec('example_3') # gamma=0.5
    answers = {
        'v': [[
            np.array([2.03125, 4.5, 0.75, 0]), # mdp
            np.array([2.03125, 3.5625, 0]), # amdp / mc*
            np.array([2.03125, 3.5625, 0]), # td
        ]],
        'q': [[
            np.array([
                [2.25, 5, 0, 0], # mdp
                [1.375, 3, 3, 0]
            ]),
            np.array([
                [2.25, 3.75, 0], # amdp
                [1.375, 3, 0],
            ]),
            np.array([
                [1.78125, 3.75, 0], # td
                [2.78125, 3, 0],
            ])
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_7():
    spec = load_spec('example_7') # gamma=0.5
    answers = {
        'v': [[
            np.array([0.25, 0.5, 1., 0.]), # mdp
            np.array([0.4, 0.5, 0.]), # amdp / mc*
            np.array([0.25, 0.125, 0.]), # td
        ]],
        'q': [[
            np.array([
                [0.25, 0.5, 1., 0.], # mdp
                [1.25, 0.5, 0., 0.]
            ]),
            np.array([
                [0.4, 0.5, 0.], # amdp
                [1., 0.5, 0.]
            ]),
            np.array([
                [0.25, 0.125, 0.], # td
                [0.85, 0.125, 0.],
            ])
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_7_memory():
    spec = load_spec('example_7')
    mem_params = get_memory(str(4))

    spec['Pi_phi_x'] = [
        np.array([
            [0., 1], # Optimal policy with memory
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ]),
    ]

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

    assert_pe_results(spec, answers, mem_params=mem_params)

def test_example_11():
    spec = load_spec('example_11')
    answers = {
        'v': [[
            np.array([1 / 7, 2 / 7, 4 / 7, 0]),
            np.array([1 / 7, 0.38095238, 0]),
            np.array([1 / 7, 2 / 7, 0]),
        ]]
    }

    assert_pe_results(spec, answers)

def test_example_13():
    spec = load_spec('example_13')
    answers = {
        'v': [[
            np.array([1 / 7, 2 / 7, 4 / 7, 0]),
            np.array([0.23809524, 2 / 7, 0]),
            np.array([1 / 7, .5 / 7, 0.]),
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
