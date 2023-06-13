import numpy as np
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import MDP, POMDP, environment
from grl.vi import po_policy_iteration, value_iteration

def test_vi():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)

    print(f"Testing value iteration on Simple Chain.")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    ground_truth_vals = spec['gamma']**np.arange(chain_length - 2, -1, -1)
    v = value_iteration(mdp)

    print(f"Value iteration values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))

def test_vi_tmaze():
    spec = environment.load_spec('tmaze_5_two_thirds_up')
    print(f"Testing value iteration on T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    v = value_iteration(mdp)

    ground_truth_vals = (4 * (spec['gamma']**np.arange(8 - 2, -1, -1))).repeat(2)

    print(f"Value iteration values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))

def test_po_pi_tmaze():
    spec = environment.load_spec('tmaze_5_two_thirds_up')
    print(f"Testing value iteration on T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])
    phi_pi = po_policy_iteration(amdp)

    print(f"Policy Iteration pi: {phi_pi}\n")

if __name__ == "__main__":
    # test_vi_tmaze()
    test_po_pi_tmaze()
