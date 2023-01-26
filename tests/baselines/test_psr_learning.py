import pytest
import numpy as np

from grl import load_spec, MDP, AbstractMDP
from grl.baselines.psr_jong import discover_tests, learn_weights

def test_learning_cheese():
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
    print(Q)

    assert len(Q) == 11

    psr_model, err = learn_weights(amdp, Q)
    # print(psr_model.pi.shape)
    # print(psr_model.pi)


    # make sure we generated enough extension tests
    assert len(psr_model.weights.keys()) == amdp.n_obs * amdp.n_actions * len(Q) 

    # make sure we generated no duplicates
    assert len(psr_model.weights.keys()) == len(set(psr_model.weights.keys()))

    # assert error is close to paper's value
    #assert round(err-0.00037, 7) == 0, f"Error was {err}, should be close to 0.00037"


def test_learning_4x3():
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

    psr_model, err = learn_weights(amdp, Q)
    # print(psr_model.pi.shape)
    # print(psr_model.pi)


    # make sure we generated enough extension tests
    assert len(psr_model.weights.keys()) == amdp.n_obs * amdp.n_actions * len(Q) 

    # make sure we generated no duplicates
    assert len(psr_model.weights.keys()) == len(set(psr_model.weights.keys()))

    # assert error is close to paper's value
    #assert round(err-0.066509, 7) == 0, f"Error was {err}, should be close to 0.066509"


def test_learning_floatreset():
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
    assert Q == tests, f"Tests found were {Q}"
    # going to use these tests for consistency?
    psr_model, err = learn_weights(amdp, Q, steps=2000000, stepsize_delta=0.5, stepsize_reduce_interval=0.05)
    # print(psr_model.pi.shape)
    # print(psr_model.pi)


    # make sure we generated enough extension tests
    #assert len(psr_model.weights.keys()) == amdp.n_obs * amdp.n_actions * len(Q) 

    # make sure we generated no duplicates
    #assert len(psr_model.weights.keys()) == len(set(psr_model.weights.keys()))

    # assert error is close to paper's value
    #assert round(err-0.066509, 7) == 0, f"Error was {err}, should be close to 0.066509"


if __name__ == "__main__":
    #test_learning_cheese()
    test_learning_floatreset()
