from argparse import Namespace
from functools import partial
from itertools import product
from typing import Union

import jax
from jax.config import config
from jax.nn import softmax
import jax.numpy as jnp
from tqdm import tqdm
from collections import namedtuple

from grl.environment import load_spec
from grl.memory import memory_cross_product, get_memory
from grl.mdp import MDP, POMDP
from grl.utils.math import normalize
from grl.utils.mdp import get_td_model, get_p_s_given_o, amdp_get_occupancy
from grl.utils.policy_eval import lstdq_lambda
from scripts.memory_gradient.check_val_grads import mem_obs_val_func
from scripts.memory_gradient.intermediate_sample_grads import mem_func

ValGradInputs = namedtuple('ValGradInputs', [
    'mem_params', 'amdp', 'pi', 'T_td', 'all_mem_grads', 'unflat_v_mem', 'all_om_grads',
    'all_prod_grads'
])

def get_init_belief(amdp: POMDP, pi: jnp.ndarray):
    amdp_occupancy = amdp_get_occupancy(pi, amdp)
    return get_p_s_given_o(amdp.phi, amdp_occupancy)

def mem_prod_val(mem_params: jnp.ndarray, amdp: POMDP, pi: jnp.ndarray, mem: int, obs: int,
                 action: int, next_mem: int, next_obs: int):
    return mem_func(mem_params, obs, action, mem, next_mem) * mem_obs_val_func(
        mem_params, amdp, pi, next_obs, next_mem)

def q_mem_val(mem_params: jnp.ndarray, amdp: POMDP, pi: jnp.ndarray, mem: int, obs: int,
              action: int):
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    v0, q0, info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=0)
    q0_unflat = q0.reshape(amdp.action_space.n, -1, mem_params.shape[-1])
    return q0_unflat[action, obs, mem]

@partial(jax.jit, static_argnames='a')
def belief_update(prev_belief: jnp.ndarray, amdp: POMDP, a: int):
    next_belief = amdp.T[a] @ prev_belief
    prob_next_o = next_belief @ amdp.phi
    return next_belief, prob_next_o

def val_grad_unroll(obs: int,
                    mem: int,
                    val_grad_inputs: Union[ValGradInputs, Namespace],
                    unrolling_steps: int = 1):
    if unrolling_steps == 0:
        return val_grad_inputs.all_om_grads[obs, mem]

    mem_params = val_grad_inputs.mem_params
    amdp = val_grad_inputs.amdp
    pi = val_grad_inputs.pi

    cumulative_mem_grads = jnp.zeros_like(mem_params)
    mem_probs = softmax(mem_params, axis=-1)

    for a in range(amdp.action_space.n):
        pi_a_o = pi[obs, a]

        if pi_a_o == 0:
            continue

        all_next_obs = jnp.zeros_like(mem_params)

        # next_belief, prob_next_o = belief_update(belief, amdp, a)

        for next_obs in range(amdp.observation_space.n):
            # next_belief = belief
            p_op_o_a = val_grad_inputs.T_td[a, obs, next_obs]
            # p_op_o_a = prob_next_o[next_obs]

            if p_op_o_a == 0:
                continue

            all_next_mems = jnp.zeros_like(mem_params)

            for next_mem in range(mem_params.shape[-1]):
                # if unrolling_steps == 1:
                #     all_next_mems += val_grad_inputs.all_prod_grads[mem, obs, a, next_mem, next_obs]
                #     continue

                cum = val_grad_inputs.unflat_v_mem[next_obs, next_mem] * \
                      val_grad_inputs.all_mem_grads[mem, obs, a, next_mem]

                recur = 0
                if mem_probs[a, obs, mem, next_mem] > 0:
                    recur = mem_probs[a, obs, mem, next_mem] * \
                            val_grad_unroll(next_obs, next_mem, val_grad_inputs, unrolling_steps - 1)
                    # recur = mem_probs[a, obs, mem, next_mem] * \
                    #         val_grad_unroll(next_obs, next_mem, next_belief, val_grad_inputs, unrolling_steps - 1)

                all_next_mems += (cum + recur)
            all_next_obs += (p_op_o_a * amdp.gamma * all_next_mems)

        cumulative_mem_grads += (pi_a_o * all_next_obs)

    return cumulative_mem_grads

def prod_val_grad(obs: int, mem: int, val_grad_inputs: ValGradInputs):
    grad_fn = jax.grad(mem_prod_val)

    mem_params = val_grad_inputs.mem_params
    amdp = val_grad_inputs.amdp
    pi = val_grad_inputs.pi

    cumulative_mem_grads = jnp.zeros_like(mem_params)
    for a in range(amdp.action_space.n):
        pi_a_o = pi[obs, a]
        if pi_a_o == 0:
            continue

        all_next_obs = jnp.zeros_like(mem_params)

        for next_obs in range(amdp.observation_space.n):
            p_op_o_a = val_grad_inputs.T_td[a, obs, next_obs]

            if p_op_o_a == 0:
                continue

            all_next_mems = jnp.zeros_like(mem_params)
            for next_mem in range(mem_params.shape[-1]):
                all_next_mems += grad_fn(mem_params, amdp, pi, mem, obs, a, next_mem, next_obs)
            all_next_obs += (p_op_o_a * amdp.gamma * all_next_mems)

        cumulative_mem_grads += (pi_a_o * all_next_obs)

    return cumulative_mem_grads

def calc_all_unrolled_val_grads(mem_params: jnp.ndarray,
                                amdp: POMDP,
                                pi: jnp.ndarray,
                                T_td: jnp.ndarray,
                                unflat_v_mem: jnp.ndarray,
                                all_mem_grads: jnp.ndarray,
                                all_om_grads: jnp.ndarray,
                                n_unrolls: int = 0):
    n_mem = mem_params.shape[-1]
    print(f"Calculating \grad v(o, m)'s after {n_unrolls} unrolls")
    all_om_n_grads = jnp.zeros((amdp.observation_space.n, n_mem) + mem_params.shape)
    val_grad_inputs = {
        'mem_params': mem_params,
        'amdp': amdp,
        'pi': pi,
        'T_td': T_td,
        'unflat_v_mem': unflat_v_mem,
        'all_mem_grads': all_mem_grads,
        'all_om_grads': all_om_grads
    }
    val_grad_inputs = Namespace(**val_grad_inputs)
    for o, m in tqdm(list(product(list(range(amdp.observation_space.n)), list(range(n_mem))))):
        all_om_n_grads = all_om_n_grads.at[o, m].set(
            val_grad_unroll(o, m, val_grad_inputs, unrolling_steps=n_unrolls))

    return all_om_n_grads

def test_unrolling(amdp: POMDP, pi: jnp.ndarray, mem_params: jnp.ndarray):
    n_mem = mem_params.shape[-1]
    mem_grad_fn = jax.grad(mem_func)
    prod_grad_fn = jax.grad(mem_prod_val)
    # q_grad_fn = jax.grad(q_mem_val)

    T_td, R_td = get_td_model(amdp, pi)
    T_td = normalize(T_td, axis=-1)

    mem_aug_pi = pi.repeat(n_mem, axis=0)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=0.)
    mem_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

    all_grads = jnp.zeros((n_mem, amdp.observation_space.n, amdp.action_space.n, n_mem) +
                          mem_params.shape)
    all_prod_grads = jnp.zeros((n_mem, amdp.observation_space.n, amdp.action_space.n, n_mem,
                                amdp.observation_space.n) + mem_params.shape)
    # all_oma_grads = jnp.zeros((amdp.action_space.n, amdp.observation_space.n, n_mem) + mem_params.shape)

    for m in range(n_mem):
        for o in range(amdp.observation_space.n):
            for a in range(amdp.action_space.n):
                # all_oma_grads = all_oma_grads.at[a, o, m].set(q_grad_fn(mem_params, amdp, pi, m, o, a))
                for next_m in range(n_mem):
                    all_grads = all_grads.at[m, o, a,
                                             next_m].set(mem_grad_fn(mem_params, o, a, m, next_m))
                    # FOR PROD
                    # for next_o in range(amdp.observation_space.n):
                    #     all_prod_grads = all_prod_grads.at[m, o, a, next_m, next_o].set(prod_grad_fn(mem_params, amdp, pi, m, o, a, next_m, next_o))

    print("Calculating base \grad v(o, m)'s")
    all_om_grads = jnp.zeros((amdp.observation_space.n, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.observation_space.n)), list(range(n_mem))))):
        all_om_grads = all_om_grads.at[o, m].set(
            jax.grad(mem_obs_val_func)(mem_params, amdp, pi, o, m))

    val_grad_inputs = ValGradInputs(mem_params, amdp, pi, T_td, all_grads, mem_v0_unflat,
                                    all_om_grads, all_prod_grads)

    # with jax.disable_jit():
    n_unrolls = 2
    print(f"Calculating \grad v(o, m)'s after {n_unrolls} unrolls")
    all_om_n_grads = jnp.zeros((amdp.observation_space.n, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.observation_space.n)), list(range(n_mem))))):
        all_om_n_grads = all_om_n_grads.at[o, m].set(
            val_grad_unroll(o, m, val_grad_inputs, unrolling_steps=n_unrolls))
    print("are they the same???")

def test_product_grad(amdp: POMDP, pi: jnp.ndarray, mem_params: jnp.ndarray):
    n_mem = mem_params.shape[-1]
    grad_fn = jax.grad(mem_func)
    T_td, R_td = get_td_model(amdp, pi)

    mem_aug_pi = pi.repeat(n_mem, axis=0)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp)
    mem_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

    all_grads = jnp.zeros((n_mem, amdp.observation_space.n, amdp.action_space.n, n_mem) +
                          mem_params.shape)

    for m in range(n_mem):
        for o in range(amdp.observation_space.n):
            for a in range(amdp.action_space.n):
                for next_m in range(n_mem):
                    all_grads = all_grads.at[m, o, a,
                                             next_m].set(grad_fn(mem_params, o, a, m, next_m))

    print("Calculating base \grad v(o, m)'s")
    all_om_grads = jnp.zeros((amdp.observation_space.n, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.observation_space.n)), list(range(n_mem))))):
        all_om_grads = all_om_grads.at[o, m].set(
            jax.grad(mem_obs_val_func)(mem_params, amdp, pi, o, m))

    val_grad_inputs = ValGradInputs(mem_params, amdp, pi, T_td, all_grads, mem_v0_unflat,
                                    all_om_grads)

    print(f"Calculating \grad v(o, m)'s after product")
    all_om_prod_grads = jnp.zeros((amdp.observation_space.n, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.observation_space.n)), list(range(n_mem))))):
        all_om_prod_grads = all_om_prod_grads.at[o, m].set(prod_val_grad(o, m, val_grad_inputs))

    print("are they the same???")

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
    mem_params = get_memory('fuzzy',
                            n_obs=amdp.observation_space.n,
                            n_actions=amdp.action_space.n,
                            leakiness=0.2)

    test_unrolling(amdp, pi, mem_params)
    # test_product_grad(amdp, pi, mem_params)
