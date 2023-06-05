import numpy as np
import jax
from jax.nn import softmax
from tqdm import trange
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import MDP, AbstractMDP
from grl.environment import load_spec
from grl.agent.analytical import AnalyticalAgent
from grl.utils.math import glorot_init

def test_policy_grad_fully_observable_tmaze():
    iterations = 5000
    spec = load_spec('tmaze_5_two_thirds_up_fully_observable')
    print(f"Testing analytical policy gradient on fully observable T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    np.random.seed(2020)
    rand_key = jax.random.PRNGKey(2020)
    pi_params = glorot_init((spec['Pi_phi'][0].shape))

    agent = AnalyticalAgent(pi_params, rand_key, pi_lr=0.01, policy_optim_alg='policy_grad')

    for it in trange(iterations):
        v_0 = agent.policy_improvement(amdp)

    learnt_pi = softmax(agent.pi_params, axis=-1)
    assert np.allclose(learnt_pi[:-3, 2], np.ones_like(learnt_pi[:-3, 2]), atol=1e-2), \
        f"Learnt policy should be go right in corridor, but is {learnt_pi[:-3]} instead."

    assert np.isclose(learnt_pi[-2, 1], 1, atol=5e-2) and np.isclose(learnt_pi[-3, 0], 1, atol=1e-2), \
        "Learnt policy should be to go up/down depending on reward."

    print(f"Learnt policy gradient policy: \n"
          f"{learnt_pi}")

def test_policy_grad_tmaze():
    iterations = 12000
    spec = load_spec('tmaze_5_two_thirds_up')
    print(f"Testing analytical policy gradient on partially observable T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    seed = 2020
    np.random.seed(seed)
    rand_key = jax.random.PRNGKey(seed)
    pi_params = glorot_init((spec['Pi_phi'][0].shape))

    agent = AnalyticalAgent(pi_params, rand_key, pi_lr=0.0025, policy_optim_alg='policy_grad')

    for it in trange(iterations):
        v_0 = agent.policy_improvement(amdp)

    learnt_pi = softmax(agent.pi_params, axis=-1)
    if np.isclose(learnt_pi[0, 2], 1, atol=1e-2):
        assert np.isclose(learnt_pi[-2, 0], 1, atol=1e-2), f"Learnt pi right: {learnt_pi[-2, 0]}"
    else:
        assert np.isclose(learnt_pi[-2, 1], 1, atol=1e-2), f"Learnt pi right: {learnt_pi[-2, 1]}"

    print(f"Learnt policy gradient policy: \n"
          f"{learnt_pi}")

if __name__ == "__main__":
    test_policy_grad_fully_observable_tmaze()
    test_policy_grad_tmaze()
