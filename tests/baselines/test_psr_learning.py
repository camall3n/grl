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

if __name__ == "__main__":
    test_learning_cheese()
