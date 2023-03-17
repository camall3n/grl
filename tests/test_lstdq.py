import jax.numpy as jnp
import numpy as np

from grl.environment import load_spec
from grl.mdp import AbstractMDP, MDP
from grl.utils.policy_eval import analytical_pe, lstdq_lambda

def test_lstdq():
    spec_name = 'tmaze_eps_hyperparams'
    lamb = 0

    spec = load_spec(spec_name, epsilon=0.1)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]

    _, mc_vals, td_vals, _ = analytical_pe(pi, amdp)

    vlstd_lambda_0, qlstd_lambda_0 = lstdq_lambda(pi, amdp, lambda_=0)
    vlstd_lambda_1, qlstd_lambda_1 = lstdq_lambda(pi, amdp, lambda_=1)

    assert np.allclose(mc_vals['v'], vlstd_lambda_1)
    assert np.allclose(mc_vals['q'], qlstd_lambda_1)

    assert np.allclose(td_vals['v'], vlstd_lambda_0)
    assert np.allclose(td_vals['q'], qlstd_lambda_0)
