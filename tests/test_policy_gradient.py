import numpy as np
from jax.nn import softmax
from tqdm import trange

from grl import MDP, AbstractMDP, PolicyEval, environment
from grl.analytical_agent import AnalyticalAgent

def test_pg_fully_observable_tmaze():
    iterations = 10000
    lr = 1
    spec = environment.load_spec('tmaze_5_two_thirds_up_fully_observable')
    print(f"Testing analytical policy gradient on fully observable T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    pi_params = np.random.normal(size=(spec['Pi_phi'][0].shape)) * np.sqrt(2)

    agent = AnalyticalAgent(pi_params)

    for it in trange(iterations):
        v_0 = agent.policy_improvement(amdp, lr)
        if it % 200 == 0:
            print(f"initial state value for iteration {it}: {v_0:.4f}")


    learnt_pi = softmax(agent.pi_params, axis=-1)
    assert np.allclose(learnt_pi[:-3, 2], np.ones_like(learnt_pi[:-3, 2]), atol=1e-2), \
        "Learnt policy is not to go right in corridor"

    assert np.isclose(learnt_pi[-2, 1], 1, atol=1e-2) and np.isclose(learnt_pi[-3, 0], 1, atol=1e-2), \
        "Learnt policy is not to go up/down depending on reward"

    print(f"Learnt policy gradient policy: \n"
          f"{learnt_pi}")

def test_pg_tmaze():
    iterations = 20000
    lr = 1
    spec = environment.load_spec('tmaze_5_two_thirds_up')
    print(f"Testing analytical policy gradient on fully observable T-Maze.")

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    pi_params = np.random.normal(size=(spec['Pi_phi'][0].shape)) * np.sqrt(2)

    agent = AnalyticalAgent(pi_params)

    for it in trange(iterations):
        v_0 = agent.policy_improvement(amdp, lr)
        if it % 1000 == 0:
            print(f"initial state value for iteration {it}: {v_0:.4f}")

    learnt_pi = softmax(agent.pi_params, axis=-1)
    if np.isclose(learnt_pi[0, 2], 1, atol=1e-2):
        assert np.isclose(learnt_pi[-2, 0], 1, atol=1e-2)
    else:
        assert np.isclose(learnt_pi[-2, 1], 1, atol=1e-2)

    print(f"Learnt policy gradient policy: \n"
          f"{learnt_pi}")

if __name__ == "__main__":
    test_pg_tmaze()
