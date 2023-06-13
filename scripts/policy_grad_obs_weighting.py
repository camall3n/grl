from functools import partial

import jax
from jax.config import config
from jax.nn import softmax
import jax.numpy as jnp

from grl.environment import load_spec
from grl.memory import memory_cross_product, get_memory
from grl.mdp import MDP, POMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.math import reverse_softmax
from grl.utils.loss import pg_objective_func

@partial(jax.jit, static_argnames='obs')
def policy_grad_objective(pi_params: jnp.ndarray, amdp: POMDP, obs: int):
    pi = softmax(pi_params, axis=-1)
    lstd_v0, lstd_q0, lstd_info = lstdq_lambda(pi, amdp, lambda_=lambda_0)
    return lstd_v0[obs]

if __name__ == "__main__":

    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.2
    lambda_0 = 0.
    lambda_1 = 1.
    n_episode_samples = int(1)
    seed = 2023

    rand_key = jax.random.PRNGKey(seed)

    spec = load_spec(spec_name,
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])

    pi = spec['Pi_phi'][0]
    pi_params = reverse_softmax(pi)
    mem_params = get_memory('fuzzy',
                            n_obs=amdp.observation_space.n,
                            n_actions=amdp.action_space.n,
                            leakiness=0.2)
    n_mem = mem_params.shape[-1]

    mem_aug_pi = pi.repeat(n_mem, axis=0)
    mem_aug_pi_params = reverse_softmax(mem_aug_pi)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    policy_grad_fn = jax.grad(policy_grad_objective)

    all_policy_grads = jnp.zeros((amdp.observation_space.n, ) + pi_params.shape)
    for o in range(amdp.observation_space.n):
        all_policy_grads = all_policy_grads.at[o].set(policy_grad_fn(pi_params, amdp, o))

    # l2 norm of all gradients
    pg_norms = jnp.sqrt(jnp.einsum('ijk->i', all_policy_grads**2))

    outs, params_grad = jax.value_and_grad(pg_objective_func, has_aux=True)(pi_params, amdp)
    pg_norms_over_obs = jnp.linalg.norm(params_grad, axis=-1)

    mem_aug_outs, mem_aug_params_grad = jax.value_and_grad(pg_objective_func,
                                                           has_aux=True)(mem_aug_pi_params,
                                                                         mem_aug_amdp)
    pg_norms_over_mem_aug_obs = jnp.linalg.norm(mem_aug_params_grad, axis=-1)

    print("done")
