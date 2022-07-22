import numpy as np

from grl.grl import load_spec, do_grad, RTOL

def test_example_7_p():
    """
    Tests that do_grad reaches the known no-discrepancy policy for example 7
    """
    spec = load_spec('example_7')

    pi_known = np.array([
        [4 / 7, 3 / 7], # known location of no discrepancy
        [1, 0],
        [1, 0],
    ])

    pi = np.array([[1., 0], [1, 0], [1, 0]])
    pi_grad = do_grad(spec, pi, 'p', no_gamma=True, lr=1)

    assert np.allclose(pi_known[0], pi_grad[0], rtol=RTOL) # just assert the red obs policy

def test_example_7_m():
    """
    Tests that do_grad reaches a known no-discrepancy memory for example 7
    """
    spec = load_spec('example_7')
    memory_start = np.array([
        [ # red
            [1., 0], # s0, s1
            [1, 0],
        ],
        [ # blue
            [0.5, 0.5],
            [1, 0],
        ],
        [ # terminal
            [1, 0],
            [1, 0],
        ],
    ])
    spec['T_mem'] = memory_start

    memory_known = np.array([
        [ # red
            [1., 0], # s0, s1
            [1, 0],
        ],
        [ # blue
            [0, 1],
            [1, 0],
        ],
        [ # terminal
            [1, 0],
            [1, 0],
        ],
    ])
    pi = np.array([
        [1., 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
    ])
    memory_grad = do_grad(spec, pi, 'm', no_gamma=True, lr=1)

    assert np.allclose(memory_known, memory_grad, atol=1e-5)
