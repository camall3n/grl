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
from grl.memory import memory_cross_product, get_memory
from grl.mdp import MDP, POMDP
from grl.utils.loss import obs_space_mem_discrep_loss
from grl.utils.mdp import get_td_model
from grl.utils.optimizer import get_optimizer
from grl.utils.policy_eval import lstdq_lambda

from scripts.memory_gradient.check_traj_grads import mem_traj_prob
from scripts.variance_calcs import collect_episodes
from scripts.memory_gradient.val_grad_unit_tests import mem_obs_val_func, calc_all_unrolled_val_grads
from scripts.memory_gradient.intermediate_sample_grads import mem_func

@partial(jax.jit, static_argnames=['obs', 'lambda_0'])
def mem_packed_v(mem_params: jnp.ndarray,
                 amdp: POMDP,
                 pi: jnp.ndarray,
                 obs: int,
                 lambda_0: float = 0.):
    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    n_mem_states = mem_params.shape[-1]
    mem_lambda_0_v_vals, mem_lambda_0_q_vals, mem_info = lstdq_lambda(pi,
                                                                      mem_aug_amdp,
                                                                      lambda_=lambda_0)

    counts_mem_aug_flat_obs = mem_info['occupancy'] @ mem_aug_amdp.phi
    # counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, pi).T  # A x OM

    counts_mem_aug = counts_mem_aug_flat_obs.reshape(-1, n_mem_states) # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_o = counts_mem_aug / denom_counts_mem_aug

    unflattened_lambda_0_v_vals = mem_lambda_0_v_vals.reshape(-1, n_mem_states)
    reformed_lambda_0_v_vals = (unflattened_lambda_0_v_vals * prob_mem_given_o).sum(axis=-1)
    # reformed_lambda_0_v_vals = r
    return reformed_lambda_0_v_vals[obs]

@partial(jax.jit, static_argnames=['obs', 'action'])
def prob_mem_over_obs(mem_params: jnp.ndarray, mem_aug_pi: jnp.ndarray, mem_aug_amdp: POMDP,
                      obs: int, mem: int):
    n_mem_states = mem_params.shape[-1]

    mem_lambda_0_v_vals, mem_lambda_0_q_vals, mem_info = lstdq_lambda(mem_aug_pi,
                                                                      mem_aug_amdp,
                                                                      lambda_=0)

    counts_mem_aug_flat_obs = mem_info['occupancy'] @ mem_aug_amdp.phi

    counts_mem_aug = counts_mem_aug_flat_obs.reshape(-1, n_mem_states) # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_o = counts_mem_aug / denom_counts_mem_aug
    return prob_mem_given_o[obs, mem]

def lambda_discrep_loss(mem_params: jnp.ndarray,
                        amdp: POMDP,
                        lstd_v1: jnp.ndarray,
                        pi: jnp.ndarray,
                        obs: int,
                        lambda_0: float = 0.):
    reformed_v0_obs_vals = mem_packed_v(mem_params, amdp, pi, obs, lambda_0=lambda_0)
    diff = reformed_v0_obs_vals - lstd_v1[obs]
    return diff**2

@partial(jax.jit, static_argnames='t')
def calc_all_mem_grads(mem_params: jnp.ndarray, init_mem_belief: jnp.ndarray,
                       all_om_val_grads: jnp.ndarray, mem_v0_unflat: jnp.ndarray, ep: dict,
                       t: int):
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

def calc_product_rule(mem_params: jnp.ndarray, mem_aug_pi: jnp.ndarray, mem_aug_amdp: POMDP,
                      mem_lstd_v0_unflat: jnp.ndarray, all_om_val_grads: jnp.ndarray,
                      lstd_v1: jnp.ndarray, obs: int):
    n_mem = mem_params.shape[-1]
    traj_grad = jnp.zeros_like(mem_params)
    val_grad = jnp.zeros_like(mem_params)
    packed_val = 0
    for mem in range(n_mem):
        p_m_given_o, grad_pm = jax.value_and_grad(prob_mem_over_obs)(mem_params, mem_aug_pi,
                                                                     mem_aug_amdp, obs, mem)
        traj_grad += mem_lstd_v0_unflat[obs, mem] * grad_pm
        val_grad += p_m_given_o * all_om_val_grads[obs, mem]
        packed_val += p_m_given_o * mem_lstd_v0_unflat[obs, mem]
    val_diff = packed_val - lstd_v1[obs]

    return (val_diff * (traj_grad + val_grad))

def filter_deterministic_tmaze_episodes(episodes: list):
    """
    Filters out tmaze episodes into
    two episodes, starting w/ goal state down and goal state up
    """
    filtered_episodes = [episodes[0]]
    first_obs = episodes[0]['obses'][0]
    for ep in episodes[1:]:
        if ep['obses'][0] != first_obs:
            filtered_episodes.append(ep)
            return filtered_episodes

def check_samples():
    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.99999
    junction_up_pi = 1.
    epsilon = 0.1
    optim_str = 'adam'
    step_size = 0.01
    lambda_0 = 0.
    lambda_1 = 1.
    n_total_episode_samples = int(100000)
    batch_size = int(32)
    n_grad_updates = int(10000)
    seed = 2024

    rand_key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    spec = load_spec(spec_name,
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])

    mem_params = get_memory('fuzzy',
                            n_obs=amdp.observation_space.n,
                            n_actions=amdp.action_space.n,
                            leakiness=0.2)

    pi = spec['Pi_phi'][0]

    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)
    n_mem = mem_params.shape[-1]

    # initialize optimizer here
    optim = get_optimizer(optim_str, step_size)
    optim_state = optim.init(mem_params)

    print(f"Sampling {n_total_episode_samples} episodes")
    all_sampled_episodes = collect_episodes(amdp,
                                            pi,
                                            n_total_episode_samples,
                                            rand_key,
                                            mem_paramses=[mem_params])

    # DEBUGGING
    # all_sampled_episodes = filter_deterministic_tmaze_episodes(all_sampled_episodes)
    # n_total_episode_samples = len(all_sampled_episodes)
    # batch_size = n_total_episode_samples

    unrolling_steps = 2
    if unrolling_steps > 0:
        T_td, R_td = get_td_model(amdp, pi)

    lstd_v1, lstd_q1, lstd_info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

    obs_space_grad_fn = jax.jit(jax.value_and_grad(obs_space_mem_discrep_loss),
                                static_argnames=[
                                    'value_type', 'q', 'error_type', 'lambda_0', 'lambda_1',
                                    'alpha', 'flip_count_prob'
                                ])
    val_grad_fn = jax.jit(jax.grad(mem_obs_val_func), static_argnames=['obs', 'mem'])
    mem_grad_fn = jax.jit(jax.grad(mem_func),
                          static_argnames=['obs', 'action', 'prev_mem', 'next_mem'])

    # DEBUGGING
    jitted_calc_product_rule = jax.jit(calc_product_rule, static_argnames=['obs'])
    mem_packed_v_grad_fn = jax.jit(jax.value_and_grad(mem_packed_v),
                                   static_argnames=['obs', 'lambda_0'])
    lambda_discrep_grad_fn = jax.jit(jax.grad(lambda_discrep_loss),
                                     static_argnames=['obs', 'lambda_0'])

    init_mem_belief = jnp.zeros(n_mem)
    init_mem_belief = init_mem_belief.at[0].set(1)

    for g in range(n_grad_updates):
        print(f"Calculating base \grad v(o, m)'s for update {g}")
        all_om_val_grads = jnp.zeros((amdp.observation_space.n, n_mem) + mem_params.shape)
        for o, m in tqdm(list(product(list(range(amdp.observation_space.n)), list(range(n_mem))))):
            all_om_val_grads = all_om_val_grads.at[o,
                                                   m].set(val_grad_fn(mem_params, amdp, pi, o, m))

        mem_aug_amdp = memory_cross_product(mem_params, amdp)
        mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi,
                                                               mem_aug_amdp,
                                                               lambda_=lambda_0)
        mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

        if unrolling_steps > 0:
            all_mem_grads = jnp.zeros((n_mem, amdp.observation_space.n, amdp.action_space.n,
                                       n_mem) + mem_params.shape)

            print(f"Calculating memory grads for update {g}")
            for o, m in tqdm(list(product(range(amdp.observation_space.n), range(n_mem)))):
                for a, next_m in product(range(amdp.action_space.n), range(n_mem)):
                    all_mem_grads = all_mem_grads.at[m, o, a, next_m].set(
                        mem_grad_fn(mem_params, o, a, m, next_m))

            all_om_val_grads = calc_all_unrolled_val_grads(mem_params,
                                                           amdp,
                                                           pi,
                                                           T_td,
                                                           mem_lstd_v0_unflat,
                                                           all_mem_grads,
                                                           all_om_val_grads,
                                                           n_unrolls=unrolling_steps)

        mem_probs = softmax(mem_params, axis=-1)

        loss, analytical_mem_grad = obs_space_grad_fn(mem_params,
                                                      mem_aug_pi,
                                                      amdp,
                                                      alpha=1.,
                                                      value_type='v')
        print(f"Analytical loss for update #{g}: {loss.item()}")

        sampled_grads = jnp.zeros_like(mem_params)
        val_grads = jnp.zeros_like(mem_params)
        traj_grads = jnp.zeros_like(mem_params)

        print(f"Sampling grads over {batch_size} episodes")
        idxes = np.random.choice(n_total_episode_samples, size=batch_size, replace=False)
        sampled_episodes = [all_sampled_episodes[idx] for idx in idxes]

        # DEBUGGING
        # sampled_episodes = all_sampled_episodes

        for ep in tqdm(sampled_episodes):

            mem_belief = init_mem_belief

            eps_sampled_grads = jnp.zeros_like(mem_params)

            for t in range(ep['actions'].shape[0]):

                action = ep['actions'][t]
                obs = ep['obses'][t]

                val_diff = calc_val_diff(mem_lstd_v0_unflat, mem_belief, lstd_v1, obs)
                val_grad, traj_grad = calc_all_mem_grads(mem_params, init_mem_belief,
                                                         all_om_val_grads, mem_lstd_v0_unflat, ep,
                                                         t)

                eps_sampled_grads += (val_diff * (traj_grad + val_grad))

                # DEBUGGING
                # eps_sampled_grads += jitted_calc_product_rule(mem_params, mem_aug_pi, mem_aug_amdp,
                #                                               mem_lstd_v0_unflat, all_om_val_grads, lstd_v1, obs)

                # repacked_mem_v_obs, grad = mem_packed_v_grad_fn(mem_params, amdp, mem_aug_pi, obs)
                # eps_sampled_grads += (repacked_mem_v_obs - lstd_v1[obs]) * grad

                # eps_sampled_grads += lambda_discrep_grad_fn(mem_params, amdp, lstd_v1, mem_aug_pi, obs)

                mem_mat = mem_probs[action, obs]
                mem_belief = mem_belief @ mem_mat

            else:
                obs = ep['obses'][-1]

                val_diff = calc_val_diff(mem_lstd_v0_unflat, mem_belief, lstd_v1, obs)
                val_grad, traj_grad = calc_all_mem_grads(mem_params, init_mem_belief,
                                                         all_om_val_grads, mem_lstd_v0_unflat, ep,
                                                         t + 1)
                eps_sampled_grads += (val_diff * (traj_grad + val_grad))

            eps_sampled_grads /= (ep['obses'].shape[0] - 1)
            sampled_grads += eps_sampled_grads

        # sampled_grads *= (1 / (2 * batch_size))
        val_grads *= (1 / batch_size)
        traj_grads *= (1 / batch_size)

        updates, optim_state = optim.update(sampled_grads, optim_state, mem_params)
        mem_params = optax.apply_updates(mem_params, updates)

    print("done")

if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')

    check_samples()
