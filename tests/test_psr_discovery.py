import pytest
import numpy as np

from grl import load_spec, pe_grad, RTOL, MDP, AbstractMDP
from grl.baselines.psr_jong import discover_tests
from grl.memory_iteration import mem_improvement

def test_discovery_tiger():
    """
    From Littman et al. (2003): We should discover 2 core tests for Tiger.
    """
    spec = load_spec('tiger')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    # assert conditions are the same as in Littman et al. (2003)
    assert amdp.n_obs == 2, f"Number of observations is {amdp.n_obs}"
    assert amdp.n_actions == 3, f"Number of actions is {amdp.n_actions}"

    Q = discover_tests(amdp)

    assert len(Q) == 2


def test_discovery_paint():
    """
    From Littman et al. (2003): We should discover 2 core tests for Paint.
    """
    spec = load_spec('paint.95')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    # assert conditions are the same as in Littman et al. (2003)
    assert amdp.n_obs == 2, f"Number of observations is {amdp.n_obs}"
    assert amdp.n_actions == 4, f"Number of actions is {amdp.n_actions}"

    Q = discover_tests(amdp)

    assert len(Q) == 2




def test_discovery_cheese():
    """
    From Littman et al. (2003): We should discover 11 core tests for Cheese Maze.
    """
    spec = load_spec('cheese.95')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    # assert conditions are the same as in Littman et al. (2003)
    assert amdp.n_obs == 7, f"Number of observations is {amdp.n_obs}"
    assert amdp.n_actions == 4, f"Number of actions is {amdp.n_actions}"

    Q = discover_tests(amdp)

    assert len(Q) == 11
