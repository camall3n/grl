import pytest
import numpy as np
from grl.mdp import MDP

@pytest.fixture()
def mdp():
    return MDP.generate(n_states=5, n_actions=10, sparsity=0.5)

def test_get_policy(mdp):
    assert np.all(mdp.get_policy(0) == np.zeros(mdp.state_space.n))
    assert np.all(mdp.get_policy(1) == np.array([1, 0, 0, 0, 0]))
    assert np.all(mdp.get_policy(mdp.action_space.n) == np.array([0, 1, 0, 0, 0]))
    assert np.all(mdp.get_policy(mdp.action_space.n + 1) == np.array([1, 1, 0, 0, 0]))
    assert np.all(mdp.get_policy(2 * mdp.action_space.n + 1) == np.array([1, 2, 0, 0, 0]))
    assert np.all(
        mdp.get_policy(mdp.action_space.n**mdp.state_space.n - 1) == (mdp.action_space.n - 1) *
        np.ones(mdp.state_space.n))

def np_get_policy(n, base, length):
    assert n < length**base
    policy = np.zeros(length, dtype=int)
    remainder = n
    for i in range(length):
        remainder, policy[i] = np.divmod(remainder, base)
        if remainder == 0:
            break
    return policy

def test_against_np(mdp):
    for _ in range(100):
        N = np.random.randint(mdp.action_space.n**mdp.state_space.n)
        x = mdp.get_policy(N)
        y = np_get_policy(N, mdp.action_space.n, mdp.state_space.n)
        assert np.all(x == y)
