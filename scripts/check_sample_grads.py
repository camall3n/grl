from itertools import product

import jax
from jax.nn import softmax
import jax.numpy as jnp
from jax.config import config
from tqdm import tqdm

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
from grl.utils.loss import obs_space_mem_discrep_loss
from grl.utils.policy_eval import lstdq_lambda

from scripts.check_traj_grads import mem_traj_prob
from scripts.variance_calcs import collect_episodes
from scripts.val_grad_unit_tests import val_grad_unroll, mem_obs_val_func


if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.2
    lambda_0 = 0.
    lambda_1 = 1.
    n_episode_samples = int(1000)
    seed = 2023

    rand_key = jax.random.PRNGKey(seed)

    spec = load_spec(spec_name,
                     # memory_id=str(mem),
                     memory_id=str('f'),
                     mem_leakiness=0.2,
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    pi = spec['Pi_phi'][0]
    mem_params = spec['mem_params']
    mem_probs = softmax(mem_params, axis=-1)

    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    n_mem = mem_params.shape[-1]

    print(f"Sampling {n_episode_samples} episodes")
    sampled_episodes = collect_episodes(amdp, pi, n_episode_samples, rand_key, mem_paramses=[mem_params])

    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=lambda_0)
    mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

    lstd_v1, lstd_q1, lstd_info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

    print("Calculating base \grad v(o, m)'s")
    all_om_val_grads = jnp.zeros((amdp.n_obs, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.n_obs)), list(range(n_mem))))):
        all_om_val_grads = all_om_val_grads.at[o, m].set(jax.grad(mem_obs_val_func)(mem_params, amdp, pi, o, m))

    obs_space_grad_fn = jax.grad(obs_space_mem_discrep_loss)
    analytical_mem_grad = obs_space_grad_fn(mem_params, mem_aug_pi, amdp)

    traj_grad_fn = jax.jit(jax.value_and_grad(mem_traj_prob), static_argnames='T')

    init_mem_belief = jnp.zeros(n_mem)
    init_mem_belief = init_mem_belief.at[0].set(1)

    sampled_grads = jnp.zeros_like(mem_params)
    print(f"Sampling grads over {n_episode_samples} episodes")
    for ep in tqdm(sampled_episodes):

        mem_belief = init_mem_belief
        for t in range(ep['actions'].shape[0]):
            action = ep['actions'][t]
            obs = ep['obses'][t]
            # mem = ep['memses'][t][0]

            traj_grad = jnp.zeros_like(mem_params)
            val_grad = jnp.zeros_like(mem_params)
            for m in range(n_mem):
                m_belief, m_traj_grad = traj_grad_fn(mem_params, init_mem_belief, m, ep, T=t)
                traj_grad += mem_lstd_v0_unflat[obs, m] * m_traj_grad

                val_grad += m_belief * all_om_val_grads[obs, m]

            val_diff = jnp.dot(mem_lstd_v0_unflat[obs], mem_belief) - lstd_v1[obs]

            sampled_grads += (1 / (2 * n_episode_samples)) * (val_diff * (traj_grad + val_grad))

            mem_mat = mem_probs[action, obs]
            mem_belief = mem_belief @ mem_mat


    print("done")
