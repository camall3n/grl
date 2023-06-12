import pytest
import numpy as np

from grl.mdp import MDP, AbstractMDP
from grl.environment import tmaze_lib

def test_slippery_tmaze():
    corridor_length = 5
    slip_prob = 0.1
    T, R, gamma, p0, phi = tmaze_lib.slippery_tmaze(corridor_length, slip_prob=slip_prob)
    mdp = MDP(T, R, p0, gamma=1.0)
    slip_tmaze = AbstractMDP(mdp, phi)

    # make sure we have proper prob. dists.
    assert np.allclose(T.sum(axis=-1), 1)

    # Now we test slip probabilities
    samples = 10000
    go_right_states = np.arange(0, slip_tmaze.state_space.n - 3)
    go_left_states = np.arange(2, slip_tmaze.state_space.n - 1)

    success_right_counts, success_left_counts = np.zeros_like(go_right_states), np.zeros_like(
        go_left_states)
    for _ in range(samples):
        for s in go_right_states:
            slip_tmaze.reset(s)
            info = slip_tmaze.step(2)[-1]
            next_s = info['state']
            success_right_counts[s] += next_s == s + 2

        for s in go_left_states:
            slip_tmaze.reset(s)
            info = slip_tmaze.step(3)[-1]
            next_s = info['state']
            success_left_counts[s - 2] += next_s == s - 2
    success_right_ratios = success_right_counts / samples
    success_left_ratios = success_left_counts / samples

    assert np.allclose(success_right_ratios, 1 - slip_prob, atol=1e-2)
    assert np.allclose(success_left_ratios, 1 - slip_prob, atol=1e-2)

@pytest.fixture()
def tmaze():
    corridor_length = 5
    T, R, gamma, p0, phi = tmaze_lib.tmaze(corridor_length)
    mdp = MDP(T, R, p0, gamma=1)
    tmaze = AbstractMDP(mdp, phi)
    return tmaze

def test_tmaze_start(tmaze):
    n_samples = 10000

    # first we test our start states
    start_counts = np.zeros(2)
    for _ in range(n_samples):
        ob, info = tmaze.reset()
        s = info['state']
        start_counts[s] += 1

        if s == 0:
            assert ob == 0
        elif s == 1:
            assert ob == 1

    assert np.all(
        np.isclose(start_counts / n_samples, np.zeros_like(start_counts) + 0.5, atol=1e-1))

def test_tmaze_noop(tmaze):
    # Now we test bumping - or self-loops/no-op actions.
    bump_ns_states = np.arange(0, tmaze.state_space.n - 3, dtype=int)
    for s in bump_ns_states:
        # test up
        tmaze.reset(s)
        tmaze.gamma
        _, r, terminal, _, info = tmaze.step(0)
        next_s = info['state']
        assert next_s == s and r == 0 and not terminal

        # test down
        tmaze.reset(s)
        _, r, terminal, _, info = tmaze.step(1)
        next_s = info['state']
        assert next_s == s and r == 0 and not terminal

    bump_east_states = np.array([tmaze.state_space.n - 2 - 1, tmaze.state_space.n - 1 - 1])
    bump_west_states = np.array([0, 1])
    for s in bump_east_states:
        tmaze.reset(s)
        _, r, terminal, _, info = tmaze.step(2)
        next_s = info['state']
        assert next_s == s and r == 0 and not terminal

    for s in bump_west_states:
        tmaze.reset(s)
        _, r, terminal, _, info = tmaze.step(3)
        next_s = info['state']
        assert next_s == s and r == 0 and not terminal

# Now we test moving
def test_tmaze_move(tmaze):
    go_right_states = np.arange(0, tmaze.state_space.n - 3)
    for s in go_right_states:
        tmaze.reset(s)
        _, r, terminal, _, info = tmaze.step(2)
        next_s = info['state']
        assert next_s == s + 2 and r == 0 and not terminal

    go_left_states = np.arange(2, tmaze.state_space.n - 1)
    for s in go_left_states:
        tmaze.reset(s)
        _, r, terminal, _, info = tmaze.step(3)
        next_s = info['state']
        assert next_s == s - 2 and r == 0 and not terminal

def test_tmaze_terminals(tmaze):
    # make sure our terminals are terminals
    for i in range(tmaze.action_space.n):
        tmaze.reset(tmaze.state_space.n - 1)
        _, r, terminal, _, info = tmaze.step(0)
        next_s = info['state']
        assert terminal and next_s == tmaze.state_space.n - 1 and r == 0

def test_tmaze_rewards(tmaze):
    # Test rewarding transitions
    top_junction = tmaze.state_space.n - 2 - 1
    bottom_junction = tmaze.state_space.n - 1 - 1

    # top rewards
    tmaze.reset(top_junction)
    _, r, terminal, _, info = tmaze.step(0)
    next_s = info['state']
    assert terminal and next_s == tmaze.state_space.n - 1 and r == 4

    tmaze.reset(top_junction)
    _, r, terminal, _, info = tmaze.step(1)
    next_s = info['state']
    assert terminal and next_s == tmaze.state_space.n - 1 and r == -0.1

    # Bottom rewards
    tmaze.reset(bottom_junction)
    _, r, terminal, _, info = tmaze.step(0)
    next_s = info['state']
    assert terminal and next_s == tmaze.state_space.n - 1 and r == -0.1

    tmaze.reset(bottom_junction)
    _, r, terminal, _, info = tmaze.step(1)
    next_s = info['state']
    assert terminal and next_s == tmaze.state_space.n - 1 and r == 4

def test_tmaze_obs(tmaze):
    all_0_obs_states = np.array([0])
    all_1_obs_states = np.array([1])
    all_2_obs_states = np.arange(2, tmaze.state_space.n - 2 - 1)
    all_3_obs_states = np.arange(tmaze.state_space.n - 2 - 1, tmaze.state_space.n - 1)

    for i, list_s in enumerate(
        [all_0_obs_states, all_1_obs_states, all_2_obs_states, all_3_obs_states]):
        for s in list_s:
            assert tmaze.observe(s) == i

    print("All tests passed for T-maze")

if __name__ == "__main__":
    test_slippery_tmaze()
