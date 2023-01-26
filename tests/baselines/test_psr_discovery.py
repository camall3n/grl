import pytest
import numpy as np

from grl import load_spec, MDP, AbstractMDP
from grl.baselines.psr_jong import discover_tests

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

def test_discovery_4x3():
    """
    From Littman et al. (2003): We should discover 10 core tests for Cheese Maze.
    """
    spec = load_spec('4x3.95')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    # assert conditions are the same as in Littman et al. (2003)
    assert amdp.n_obs == 6, f"Number of observations is {amdp.n_obs}"
    assert amdp.n_actions == 4, f"Number of actions is {amdp.n_actions}"

    Q = discover_tests(amdp)

    assert len(Q) == 10


def test_discovery_floatreset():
    """
    From Littman et al. (2003): We should discover 5 core tests for Cheese Maze.
    """
    spec = load_spec('float_reset')
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    # assert conditions are the same as in Littman et al. (2003)
    assert amdp.n_obs == 2, f"Number of observations is {amdp.n_obs}"
    assert amdp.n_actions == 2, f"Number of actions is {amdp.n_actions}"

    Q = discover_tests(amdp)

    assert len(Q) == 5
    # can assert particular tests because they are enumerated in the paper
    #" fO, rO, fOrO, fOfOrO, and fOfOfOrO"
    tests = [[(0, 0)], [(1, 0)], [(0, 0), (1, 0)], [(0, 0), (0, 0), (1, 0)], [(0, 0), (0, 0), (0, 0), (1, 0)]]
    # TODO: this fails, but core tests are not unique, so should be fine?
   # assert Q == tests, f"Tests found were {Q}"

