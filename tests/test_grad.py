import numpy as np

from grl.grl import PolicyEval, load_spec, MDP, AbstractMDP, do_grad


def test_grad():
    """
    Tests that do_grad reaches the known no-discrepancy policy for example 7
    """
    spec = load_spec('example_7')
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])
    pe = PolicyEval(amdp)

    pi_known = np.array([
            [4/7, 3/7], # known location of no discrepancy
            [1, 0],
            [1, 0],
    ])

    pi = np.array([
            [1., 0],
            [1, 0],
            [1, 0]
    ])
    pi_grad = do_grad(pe, pi, True, 1)

    assert np.allclose(pi_known[0], pi_grad[0], rtol=1e-4) # just assert the red obs policy
