import numpy as np
from jax.nn import softmax

from grl import load_spec, do_grad, RTOL

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
    pi_grad = do_grad(spec, pi, 'p', lr=1e-2)

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
            # [0.99, 0.01],
            [1, 0],
            [1, 0],
        ],
        [ # terminal
            [1, 0],
            [1, 0],
        ],
    ])
    memory_start[memory_start == 1] -= 1e-5
    memory_start[memory_start == 0] += 1e-5
    memory_start[1, 0, 0] -= 1e-2
    memory_start[1, 0, 1] += 1e-2
    spec['mem_params'] = np.log(np.array([memory_start, memory_start]))

    memory_end = np.array([
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
    # Policy is all up, so only up action memory is changed
    memory_end = np.array([memory_end, memory_start])

    pi = np.array([
        [1., 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
    ])
    memory_grad = do_grad(spec, pi, 'm', lr=1)

    assert np.allclose(memory_end, softmax(memory_grad, axis=-1), atol=1e-2)
