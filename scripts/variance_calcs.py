import numpy as np
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP

if __name__ == "__main__":
    spec_name = 'tmaze_5_two_thirds_up'
    n_step = float('inf')
    spec = load_spec(spec_name)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    n_step_sample_v = np.zeros(amdp.n_obs)
    n_step_sample_q = np.zeros((amdp.n_actions, amdp.n_obs))



