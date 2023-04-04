"""
In this script, we try and figure out how to calculate p(m | o, a)
"""
from jax.config import config
import jax.numpy as jnp

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
from grl.utils.mdp import amdp_get_occupancy
from grl.utils.policy_eval import lstdq_lambda

if __name__ == "__main__":

    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    # spec_name = "simple_chain"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 2 / 3
    epsilon = 0.1
    lambda_ = 1.

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
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    mem_aug_mdp = memory_cross_product(mem_params, amdp)

    lstd_v0, lstd_q0, _ = lstdq_lambda(pi, amdp, lambda_=lambda_)
    mem_lstd_v0, mem_lstd_q0, _ = lstdq_lambda(mem_aug_pi, mem_aug_mdp, lambda_=lambda_)

    undisc_mdp = MDP(amdp.T, amdp.R, amdp.p0, lambda_)
    undisc_amdp = AbstractMDP(undisc_mdp, amdp.phi)

    undisc_mem_aug_amdp = memory_cross_product(mem_params, undisc_amdp)

    counts_mem_aug_flat_obs = amdp_get_occupancy(mem_aug_pi, undisc_mem_aug_amdp) @ undisc_mem_aug_amdp.phi
    counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, mem_aug_pi).T  # A x OM

    counts_mem_aug = counts_mem_aug_flat.reshape(amdp.n_actions, -1, 2)  # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_oa = counts_mem_aug / denom_counts_mem_aug

    mem_lstd_q0_unflat = mem_lstd_q0.reshape(amdp.n_actions, -1, mem_params.shape[-1])
    reformed_q0 = (mem_lstd_q0_unflat * prob_mem_given_oa).sum(axis=-1)
    diff = reformed_q0 - lstd_q0
    print("")


