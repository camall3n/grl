import numpy as np


from grl import MDP, environment
from grl.td_lambda import td_lambda


def test_td_lambda():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)
    n_episodes = 2000

    print(f"Testing TD(lambda) on Simple Chain over {n_episodes} episodes")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    policies = spec['Pi_phi']

    ground_truth_vals = spec['gamma'] ** np.arange(chain_length - 2, -1, -1)

    v, q = td_lambda(
        mdp,
        policies[0],
        lambda_=1,
        alpha=0.01,
        n_episodes=n_episodes,
    )

    print(f"Calculated values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))

