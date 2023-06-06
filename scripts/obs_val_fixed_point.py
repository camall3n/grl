import jax
from jax.config import config
import jax.numpy as jnp

from grl.environment import load_spec
from grl.memory import memory_cross_product, get_memory
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.mdp import get_td_model

def n_step_bellman_update(obs: int,
                          T_td: jnp.ndarray,
                          R_td: jnp.ndarray,
                          v: jnp.ndarray,
                          pi: jnp.ndarray,
                          gamma: float = 0.9,
                          depth: int = 1):
    if depth == 0:
        return v[obs]

    n_obs = v.shape[0]
    n_actions = pi.shape[-1]

    new_v = 0

    for a in range(n_actions):
        pi_a_o = pi[obs, a]

        if pi_a_o == 0:
            continue

        for next_obs in range(n_obs):
            p_op_o_a = T_td[a, obs, next_obs]

            if p_op_o_a == 0:
                continue

            update = pi_a_o * p_op_o_a * (
                R_td[a, obs, next_obs] +
                gamma * n_step_bellman_update(next_obs, T_td, R_td, v, pi, gamma, depth - 1))
            new_v += update

    return new_v

if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.2
    lambda_0 = 0.
    lambda_1 = 1.
    n_episode_samples = int(10000)
    seed = 2022

    rand_key = jax.random.PRNGKey(seed)

    spec = load_spec(
        spec_name,
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
    mem_params = get_memory('f', n_obs=amdp.n_obs, n_actions=amdp.n_actions, leakiness=0.2)
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    v0, q0, lstd_info = lstdq_lambda(pi, amdp, lambda_=lambda_0)

    T_td, R_td = get_td_model(amdp, pi)

    n_step = 4
    new_v = jnp.zeros_like(v0)
    for o in range(amdp.n_obs):
        new_v_o = n_step_bellman_update(o, T_td, R_td, v0, pi, gamma=amdp.gamma, depth=n_step)
        new_v = new_v.at[o].set(new_v_o)

    print("done")
