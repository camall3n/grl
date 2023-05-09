from itertools import product
from functools import partial

import jax
from jax.nn import softmax
import jax.numpy as jnp
from jax.config import config
import numpy as np
import optax
from tqdm import tqdm

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
from grl.utils.loss import obs_space_mem_discrep_loss
from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o
from grl.utils.optimizer import get_optimizer
from grl.utils.policy_eval import lstdq_lambda

from scripts.check_traj_grads import mem_traj_prob
from scripts.variance_calcs import collect_episodes
from scripts.val_grad_unit_tests import val_grad_unroll, mem_obs_val_func

@jax.jit
def mem_packed_v(mem_params: jnp.ndarray, amdp: AbstractMDP, pi: jnp.ndarray, obs: int):
    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    n_mem_states = mem_params.shape[-1]
    mem_lambda_0_v_vals, mem_lambda_0_q_vals, mem_info = lstdq_lambda(pi, mem_aug_amdp, lambda_=lambda_0)

    counts_mem_aug_flat_obs = mem_info['occupancy'] @ mem_aug_amdp.phi
    # counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, pi).T  # A x OM

    counts_mem_aug = counts_mem_aug_flat_obs.reshape(-1, n_mem_states)  # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_o = counts_mem_aug / denom_counts_mem_aug

    unflattened_lambda_0_v_vals = mem_lambda_0_v_vals.reshape(-1, n_mem_states)
    reformed_lambda_0_v_vals = (unflattened_lambda_0_v_vals * prob_mem_given_o).sum(axis=-1)
    # reformed_lambda_0_v_vals = r
    return reformed_lambda_0_v_vals[obs]

@partial(jax.jit, static_argnames=['obs', 'action'])
def prob_mem_over_obs(mem_params: jnp.ndarray, mem_aug_pi: jnp.ndarray, mem_aug_amdp: AbstractMDP,
                          obs: int, mem: int):
    n_mem_states = mem_params.shape[-1]

    mem_lambda_0_v_vals, mem_lambda_0_q_vals, mem_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=0)

    counts_mem_aug_flat_obs = mem_info['occupancy'] @ mem_aug_amdp.phi
    # counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, mem_aug_pi).T  # A x OM

    # counts_mem_aug = counts_mem_aug_flat.reshape(mem_aug_amdp.n_actions, -1, n_mem_states)  # A x O x M
    counts_mem_aug = counts_mem_aug_flat_obs.reshape(-1, n_mem_states)  # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_o = counts_mem_aug / denom_counts_mem_aug
    return prob_mem_given_o[obs, mem]

@partial(jax.jit, static_argnames='t')
def calc_all_mem_grads(mem_params: jnp.ndarray, init_mem_belief: jnp.ndarray,
                       all_om_val_grads: jnp.ndarray, mem_v0_unflat: jnp.ndarray,
                       ep: dict, t: int):
    traj_grad_fn = jax.value_and_grad(mem_traj_prob)

    traj_grad = jnp.zeros_like(mem_params)
    val_grad = jnp.zeros_like(mem_params)
    o = ep['obses'][t]

    for m in range(mem_params.shape[-1]):
        m_belief, m_traj_grad = traj_grad_fn(mem_params, init_mem_belief, m, ep, T=t)

        traj_grad += (mem_v0_unflat[o, m] * m_traj_grad)

        val_grad += m_belief * all_om_val_grads[o, m]

    return val_grad, traj_grad

@partial(jax.jit, static_argnames='obs')
def calc_val_diff(v0_unflat: jnp.ndarray, mem_belief: jnp.ndarray, v1: jnp.ndarray, obs: int):
    return jnp.dot(v0_unflat[obs], mem_belief) - v1[obs]


def check_samples(mem_params: jnp.ndarray, pi: jnp.ndarray, amdp: AbstractMDP, rand_key: jax.random.PRNGKey):
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)
    n_mem = mem_params.shape[-1]

    # initialize optimizer here
    optim = get_optimizer(optim_str, step_size)
    optim_state = optim.init(mem_params)

    print(f"Sampling {n_total_episode_samples} episodes")
    all_sampled_episodes = collect_episodes(amdp, pi, n_total_episode_samples, rand_key, mem_paramses=[mem_params])

    lstd_v1, lstd_q1, lstd_info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

    obs_space_grad_fn = jax.value_and_grad(obs_space_mem_discrep_loss)

    # DEBUGGING
    prob_mem_over_obs_grad_fn = jax.value_and_grad(prob_mem_over_obs)
    mem_packed_v_grad_fn = jax.grad(mem_packed_v)

    init_mem_belief = jnp.zeros(n_mem)
    init_mem_belief = init_mem_belief.at[0].set(1)

    for g in range(n_grad_updates):
        print(f"Calculating base \grad v(o, m)'s for update {g}")
        all_om_val_grads = jnp.zeros((amdp.n_obs, n_mem) + mem_params.shape)
        for o, m in tqdm(list(product(list(range(amdp.n_obs)), list(range(n_mem))))):
            all_om_val_grads = all_om_val_grads.at[o, m].set(jax.grad(mem_obs_val_func)(mem_params, amdp, pi, o, m))

        mem_aug_amdp = memory_cross_product(mem_params, amdp)
        mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=lambda_0)
        mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

        mem_probs = softmax(mem_params, axis=-1)
        loss, analytical_mem_grad = obs_space_grad_fn(mem_params, mem_aug_pi, amdp, alpha=0., value_type='v')
        print(f"Analytical loss for update #{g}: {loss.item()}")

        sampled_grads = jnp.zeros_like(mem_params)
        val_grads = jnp.zeros_like(mem_params)
        traj_grads = jnp.zeros_like(mem_params)
        print(f"Sampling grads over {n_episodes_per_update} episodes")
        idxes = np.random.choice(n_total_episode_samples, size=n_episodes_per_update, replace=False)
        sampled_episodes = [all_sampled_episodes[idx] for idx in idxes]

        for ep in tqdm(sampled_episodes):

            mem_belief = init_mem_belief
            for t in range(ep['actions'].shape[0]):

                action = ep['actions'][t]
                obs = ep['obses'][t]

                val_diff = calc_val_diff(mem_lstd_v0_unflat, mem_belief, lstd_v1, obs)
                # val_grad, traj_grad = calc_all_mem_grads(mem_params, init_mem_belief, all_om_val_grads,
                #                                          mem_lstd_v0_unflat, ep, t)

                # DEBUGGING
                # traj_grad = jnp.zeros_like(mem_params)
                # val_grad = jnp.zeros_like(mem_params)
                # for mem in range(n_mem):
                #     p_m_given_oa, grad_pm = prob_mem_over_obs_grad_fn(mem_params, mem_aug_pi, mem_aug_amdp, obs, mem)
                #     traj_grad += mem_lstd_v0_unflat[obs, mem] * grad_pm
                #     val_grad += p_m_given_oa * all_om_val_grads[obs, mem]
                #
                # val_grads += val_grad
                # traj_grads += traj_grad

                # sampled_grads += (val_diff * (traj_grad + val_grad))
                sampled_grads += (val_diff * mem_packed_v_grad_fn(mem_params, amdp, mem_aug_pi, obs))
                # sampled_grads += (1 / (2 * n_episodes_per_update))

                mem_mat = mem_probs[action, obs]
                mem_belief = mem_belief @ mem_mat


        sampled_grads *= (1 / (2 * n_episodes_per_update))
        val_grads *= (1 / n_episodes_per_update)
        traj_grads *= (1 / n_episodes_per_update)

        updates, optim_state = optim.update(sampled_grads, optim_state, mem_params)
        mem_params = optax.apply_updates(mem_params, updates)

    print("done")


if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.0
    optim_str = 'adam'
    step_size = 0.01
    lambda_0 = 0.
    lambda_1 = 1.
    n_total_episode_samples = int(1e3)
    n_episodes_per_update = int(100)
    n_grad_updates = int(1000)
    seed = 2023

    rand_key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

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

    check_samples(mem_params, pi, amdp, rand_key)
