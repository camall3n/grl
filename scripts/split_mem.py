"""
In this script, we try and figure out how to calculate p(m | o, a)
"""

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import lstdq_lambda

if __name__ == "__main__":

    spec_name = "tmaze_eps_hyperparams"
    # spec_name = "simple_chain"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 2 / 3
    epsilon = 0.0
    lambda_ = 0.9999

    spec = load_spec(spec_name,
                     memory_id=str(0),
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    pi = spec['Pi_phi'][0]
    mem_params = spec['mem_params']

    mem_aug_mdp = memory_cross_product(mem_params, amdp)

    lstd_v0, lstd_q0, _ = lstdq_lambda(pi.repeat(2, axis=0), mem_aug_mdp, lambda_=lambda_)
