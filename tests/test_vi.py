
import numpy as np

from grl import MDP, AbstractMDP, PolicyEval, environment
from grl.vi import value_iteration

def test_vi():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)

    print(f"Testing value iteration on Simple Chain.")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    ground_truth_vals = spec['gamma'] ** np.arange(chain_length - 2, -1, -1)
    v = value_iteration(mdp.T, mdp.R, mdp.gamma)

    print(f"Value iteration values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))

def test_vi_tmaze():
    spec = environment.load_spec('tmaze_5_two_thirds_up')
    print(f"Testing value iteration on T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    v = value_iteration(mdp.T, mdp.R, mdp.gamma)

    ground_truth_vals = (4 * (spec['gamma'] ** np.arange(8 - 2, -1, -1))).repeat(2)

    print(f"Value iteration values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))

if __name__ == "__main__":
    test_vi_tmaze()

    # We see what policy we get when we learn with obs only.
    # Oh no - this doesn't actually work!
    # no guarantees for memoryless policy
    spec = environment.load_spec('tmaze_5_two_thirds_up')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    amdp = AbstractMDP(mdp, spec['phi'])
    pe = PolicyEval(amdp)