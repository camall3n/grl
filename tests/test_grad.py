import numpy as np
from jax import random
from jax.nn import softmax
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import load_spec, RTOL
from grl.agent.analytical import AnalyticalAgent
from grl.memory_iteration import memory_iteration
from grl.mdp import MDP, POMDP
from grl.utils.math import reverse_softmax

def test_example_7_p():
    """
    Tests that pe_grad reaches the known no-discrepancy policy for example 7
    """
    spec = load_spec('example_7')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])
    rand_key = random.PRNGKey(2020)

    pi_known = np.array([
        [4 / 7, 3 / 7], # known location of no discrepancy
        [1, 0],
        [1, 0],
    ])

    pi = np.array([[1., 0], [1, 0], [1, 0]])
    agent = AnalyticalAgent(reverse_softmax(pi),
                            rand_key,
                            error_type='l2',
                            value_type='q',
                            policy_optim_alg='discrep_min',
                            pi_lr=0.001,
                            alpha=1.)
    info, agent = memory_iteration(agent,
                                   amdp,
                                   mi_iterations=1,
                                   mi_per_step=0,
                                   pi_per_step=int(1e3),
                                   init_pi_improvement=False)
    final_lambda_discrep_min_policy = softmax(info['final_discrep_min_pi_params'], axis=-1)

    assert np.allclose(pi_known[0], final_lambda_discrep_min_policy[0],
                       rtol=RTOL) # just assert the red obs policy

def test_example_7_m():
    """
    Tests that pe_grad reaches a known no-discrepancy memory for example 7
    """
    spec = load_spec('example_7')
    rand_key = random.PRNGKey(2020)

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
    mem_params = np.log(np.array([memory_start, memory_start]))

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])

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
        [1, 0],
        [1, 0],
        [1, 0],
    ])
    agent = AnalyticalAgent(
        reverse_softmax(pi),
        rand_key,
        optim_str='sgd',
        mem_params=mem_params,
        error_type='l2',
        value_type='q',
        # mi_lr=0.001,
        alpha=1.)
    info, agent = memory_iteration(agent,
                                   amdp,
                                   mi_iterations=1,
                                   pi_per_step=0,
                                   mi_per_step=int(5e4),
                                   init_pi_improvement=False)

    assert np.allclose(memory_end, agent.memory, atol=1e-2)

if __name__ == "__main__":
    test_example_7_m()
