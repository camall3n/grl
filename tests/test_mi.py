import numpy as np
from jax import random
from jax.nn import softmax
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import load_spec, pe_grad, MDP, AbstractMDP
from grl.agents.analytical import AnalyticalAgent
from grl.memory_iteration import mem_improvement

def test_example_7_mi():
    """
    Tests that pe_grad reaches a known no-discrepancy memory for example 7
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

    # we initialize our agent here
    pi += 1e-20
    pi_params = np.log(pi)
    rand_key = random.PRNGKey(2022)

    agent = AnalyticalAgent(pi_params,
                            rand_key,
                            mem_params=spec['mem_params'],
                            policy_optim_alg='pi',
                            epsilon=0)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    mem_loss = mem_improvement(agent, amdp, lr=1, iterations=int(1e5))
    memory_grad, _ = pe_grad(spec, pi, 'm', lr=1)

    assert np.allclose(memory_end, softmax(memory_grad, axis=-1), atol=1e-2)
    assert np.allclose(memory_end, agent.memory, atol=1e-2)

if __name__ == "__main__":
    test_example_7_mi()
