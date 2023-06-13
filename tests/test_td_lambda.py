import numpy as np
from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl import MDP, POMDP, environment
from grl.agent.td_lambda import run_td_lambda_on_mdp
from grl.utils.policy_eval import analytical_pe

def test_td_lambda():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)
    n_episodes = 2000

    print(f"Testing TD(lambda) on Simple Chain over {n_episodes} episodes")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    policies = spec['Pi_phi']

    ground_truth_vals = spec['gamma']**np.arange(chain_length - 2, -1, -1)

    v, q = run_td_lambda_on_mdp(
        mdp,
        policies[0],
        lambda_=1,
        alpha=0.01,
        n_episodes=n_episodes,
    )

    print(f"Calculated values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))

    print(f"Testing analytical solutions on Simple Chain")
    amdp = POMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]

    mdp_vals, mc_vals, td_vals, _ = analytical_pe(pi, amdp)

    assert np.all(np.isclose(mdp_vals['v'][:-1], ground_truth_vals)) and np.all(
        np.isclose(mdp_vals['q'][0][:-1], ground_truth_vals))
    assert np.all(np.isclose(mc_vals['v'][:-1], ground_truth_vals)) and np.all(
        np.isclose(mc_vals['q'][0][:-1], ground_truth_vals))
    assert np.all(np.isclose(td_vals['v'][:-1], ground_truth_vals)) and np.all(
        np.isclose(td_vals['q'][0][:-1], ground_truth_vals))
