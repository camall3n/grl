from functools import partial
from itertools import product

import jax
from jax.config import config
from jax.nn import softmax
import jax.numpy as jnp
from tqdm import tqdm

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
from grl.utils.mdp import functional_create_td_model, get_p_s_given_o, functional_get_occupancy
from grl.utils.policy_eval import lstdq_lambda
from scripts.check_val_and_traj_grads import mem_obs_val_func
from scripts.intermediate_sample_grads import mem_func

@jax.jit
def get_td_model(amdp: AbstractMDP, pi: jnp.ndarray):
    pi_state = amdp.phi @ pi
    occupancy = functional_get_occupancy(pi_state, amdp)

    p_pi_of_s_given_o = get_p_s_given_o(amdp.phi, occupancy)
    return functional_create_td_model(p_pi_of_s_given_o, amdp)

def mem_prod_val(mem_params: jnp.ndarray, amdp: AbstractMDP, pi: jnp.ndarray,
                 mem: int, obs: int, action: int, next_mem: int, next_obs: int):
    return mem_func(mem_params, obs, action, mem, next_mem) * mem_obs_val_func(mem_params, amdp, pi, next_obs, next_mem)

# def mem_prod_one_step_val(mem_params: jnp.ndarray, amdp: AbstractMDP, pi: jnp.ndarray,
#                  r: float, mem: int, obs: int, action: int, next_mem: int, next_obs: int):
#     return mem_func(mem_params, obs, action, mem, next_mem) * (r + mem_obs_val_func(mem_params, amdp, pi, next_obs, next_mem))

# @partial(jax.jit, static_argnames=['obs', 'mem', 'unrolling_steps'])
def fixed_val_grad(obs: int, mem: int, mem_params: jnp.ndarray,
                   amdp: AbstractMDP, pi: jnp.ndarray,
                   td_T: jnp.ndarray, all_mem_grads: jnp.ndarray,
                   unflat_v_mem: jnp.ndarray, all_om_grads: jnp.ndarray,
                   unrolling_steps: int = 1):
    if unrolling_steps == 0:
        return all_om_grads[obs, mem]

    cumulative_mem_grads = jnp.zeros_like(mem_params)
    mem_probs = softmax(mem_params, axis=-1)
    for a in range(amdp.n_actions):
        pi_a_o = pi[obs, a]

        if pi_a_o == 0:
            continue

        all_next_obs = jnp.zeros_like(mem_params)

        for next_obs in range(amdp.n_obs):
            p_op_o_a = td_T[a, obs, next_obs]

            if p_op_o_a == 0:
                continue

            all_next_mems = jnp.zeros_like(mem_params)

            for next_mem in range(mem_params.shape[-1]):
                cum = unflat_v_mem[next_obs, next_mem] * all_mem_grads[mem, obs, a, next_mem]

                recur = 0
                if mem_probs[a, obs, mem, next_mem] > 0:
                    recur = mem_probs[a, obs, mem, next_mem] * \
                            fixed_val_grad(next_obs, next_mem, mem_params, amdp, pi, td_T, all_mem_grads, unflat_v_mem,
                                           all_om_grads, unrolling_steps - 1)

                all_next_mems += (cum + recur)
            all_next_obs += (p_op_o_a * amdp.gamma * all_next_mems)

        cumulative_mem_grads += (pi_a_o * all_next_obs)

    return cumulative_mem_grads

@partial(jax.jit, static_argnames=['obs', 'mem'])
def prod_val_grad(obs: int, mem: int, mem_params: jnp.ndarray,
                   amdp: AbstractMDP, pi: jnp.ndarray,
                   td_T: jnp.ndarray):
    grad_fn = jax.grad(mem_prod_val)

    cumulative_mem_grads = jnp.zeros_like(mem_params)
    for a in range(amdp.n_actions):
        pi_a_o = pi[obs, a]
        all_next_obs = jnp.zeros_like(mem_params)

        for next_obs in range(amdp.n_obs):
            p_op_o_a = td_T[a, obs, next_obs]

            all_next_mems = jnp.zeros_like(mem_params)
            for next_mem in range(mem_params.shape[-1]):
                all_next_mems += grad_fn(mem_params, amdp, pi, mem, obs, a, next_mem, next_obs)
            all_next_obs += (p_op_o_a * amdp.gamma * all_next_mems)


        cumulative_mem_grads += (pi_a_o * all_next_obs)

    return cumulative_mem_grads

def test_unrolling(amdp: AbstractMDP, pi: jnp.ndarray, mem_params: jnp.ndarray):
    n_mem = mem_params.shape[-1]
    grad_fn = jax.grad(mem_func)
    T_td, R_td = get_td_model(amdp, pi)

    mem_aug_pi = pi.repeat(n_mem, axis=0)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp)
    mem_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

    all_grads = jnp.zeros((n_mem, amdp.n_obs, amdp.n_actions, n_mem) + mem_params.shape)

    for m in range(n_mem):
        for o in range(amdp.n_obs):
            for a in range(amdp.n_actions):
                for next_m in range(n_mem):
                    all_grads = all_grads.at[m, o, a, next_m].set(grad_fn(mem_params, o, a, m, next_m))

    print("Calculating base \grad v(o, m)'s")
    all_om_grads = jnp.zeros((amdp.n_obs, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.n_obs)), list(range(n_mem))))):
        all_om_grads = all_om_grads.at[o, m].set(jax.grad(mem_obs_val_func)(mem_params, amdp, pi, o, m))

    n_unrolls = 4
    print(f"Calculating \grad v(o, m)'s after {n_unrolls} unrolls")
    all_om_n_grads = jnp.zeros((amdp.n_obs, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.n_obs)), list(range(n_mem))))):
        all_om_n_grads = all_om_n_grads.at[o, m].set(fixed_val_grad(o, m, mem_params, amdp, pi, T_td, all_grads, mem_v0_unflat,
                                                                    all_om_grads, unrolling_steps=n_unrolls))
    print("are they the same???")

def test_product_grad(amdp: AbstractMDP, pi: jnp.ndarray, mem_params: jnp.ndarray):
    n_mem = mem_params.shape[-1]
    grad_fn = jax.grad(mem_func)
    T_td, R_td = get_td_model(amdp, pi)

    mem_aug_pi = pi.repeat(n_mem, axis=0)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp)
    mem_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

    all_grads = jnp.zeros((n_mem, amdp.n_obs, amdp.n_actions, n_mem) + mem_params.shape)

    for m in range(n_mem):
        for o in range(amdp.n_obs):
            for a in range(amdp.n_actions):
                for next_m in range(n_mem):
                    all_grads = all_grads.at[m, o, a, next_m].set(grad_fn(mem_params, o, a, m, next_m))

    print("Calculating base \grad v(o, m)'s")
    all_om_grads = jnp.zeros((amdp.n_obs, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.n_obs)), list(range(n_mem))))):
        all_om_grads = all_om_grads.at[o, m].set(fixed_val_grad(o, m, mem_params, amdp, pi, T_td, all_grads, mem_v0_unflat,
                                                                unrolling_steps=0))
    print(f"Calculating \grad v(o, m)'s after product")
    all_om_prod_grads = jnp.zeros((amdp.n_obs, n_mem) + mem_params.shape)
    for o, m in tqdm(list(product(list(range(amdp.n_obs)), list(range(n_mem))))):
        all_om_prod_grads = all_om_prod_grads.at[o, m].set(prod_val_grad(o, m, mem_params, amdp, pi, T_td))
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

    test_unrolling(amdp, pi, mem_params)
    # test_product_grad(amdp, pi, mem_params)

