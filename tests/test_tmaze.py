import numpy as np

from grl.mdp import MDP, AbstractMDP
from grl.environment.tmaze_lib import tmaze

def test_tmaze():
    corridor_length = 5
    T, R, gamma, p0, phi = tmaze(corridor_length)
    mdp = MDP(T, R, p0, gamma=gamma)
    pomdp = AbstractMDP(mdp, phi)

    n_samples = 10000

    # first we test our start states
    start_counts = np.zeros(2)
    for _ in range(n_samples):
        s = np.random.choice(pomdp.n_states, p=pomdp.p0)
        start_counts[s] += 1

        ob = pomdp.observe(s)
        if s == 0:
            assert ob == 0
        elif s == 1:
            assert ob == 1

    assert np.all(
        np.isclose(start_counts / n_samples, np.zeros_like(start_counts) + 0.5, atol=1e-1))

    # Now we test bumping
    bump_ns_states = np.arange(0, mdp.n_states - 3).astype(int)
    for s in bump_ns_states:
        # test up
        next_s, r, done = pomdp.step(s, 0, 1)
        assert next_s == s and r == 0 and not done

        # test down
        next_s, r, done = pomdp.step(s, 2, 1)
        assert next_s == s and r == 0 and not done

    bump_east_states = np.array([mdp.n_states - 2 - 1, mdp.n_states - 1 - 1])
    bump_west_states = np.array([0, 1])
    for s in bump_east_states:
        next_s, r, done = pomdp.step(s, 1, 1)
        assert next_s == s and r == 0 and not done

    for s in bump_west_states:
        next_s, r, done = pomdp.step(s, 3, 1)
        assert next_s == s and r == 0 and not done

    go_right_states = np.arange(0, mdp.n_states - 3)
    for s in go_right_states:
        next_s, r, done = pomdp.step(s, 1, 1)
        assert next_s == s + 2 and r == 0 and not done

    go_left_states = np.arange(2, mdp.n_states - 1)
    for s in go_left_states:
        next_s, r, done = pomdp.step(s, 3, 1)
        assert next_s == s - 2 and r == 0 and not done

    # make sure our terminals are terminals
    for i in range(mdp.n_actions):
        next_s, r, done = pomdp.step(mdp.n_states - 1, 0, 0)
        assert done and next_s == mdp.n_states - 1 and r == 0

    # Test rewarding transitions
    top_junction = mdp.n_states - 2 - 1
    bottom_junction = mdp.n_states - 1 - 1

    # top rewards
    next_s, r, done = pomdp.step(top_junction, 0, 0)
    assert done and next_s == mdp.n_states - 1 and r == 4

    next_s, r, done = pomdp.step(top_junction, 2, 0)
    assert done and next_s == mdp.n_states - 1 and r == -0.1

    # Bottom rewards
    next_s, r, done = pomdp.step(bottom_junction, 0, 0)
    assert done and next_s == mdp.n_states - 1 and r == -0.1

    next_s, r, done = pomdp.step(bottom_junction, 2, 0)
    assert done and next_s == mdp.n_states - 1 and r == 4

    all_0_obs_states = np.array([0])
    all_1_obs_states = np.array([1])
    all_2_obs_states = np.arange(2, mdp.n_states - 2 - 1)
    all_3_obs_states = np.arange(mdp.n_states - 2 - 1, mdp.n_states - 1)

    for i, list_s in enumerate(
        [all_0_obs_states, all_1_obs_states, all_2_obs_states, all_3_obs_states]):
        for s in list_s:
            assert pomdp.observe(s) == i

    print("All tests passed for T-maze")
