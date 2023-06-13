from functools import partial
import jax
from jax.config import config
from jax.nn import softmax
import jax.numpy as jnp
import numpy as np

from grl.environment import load_spec
from grl.memory import memory_cross_product, get_memory
from grl.mdp import MDP, POMDP
from grl.utils.policy_eval import lstdq_lambda

from scripts.variance_calcs import collect_episodes

def mem_traj_prob(mem_params: jnp.ndarray,
                  init_mem_belief: jnp.ndarray,
                  mem: int,
                  episode: dict,
                  T: int = None):
    mem_probs = softmax(mem_params, axis=-1)

    if T is None:
        T = episode['actions'].shape[0]

    mem_belief = init_mem_belief
    for t in range(T):
        action = episode['actions'][t]
        obs = episode['obses'][t]

        mem_mat = mem_probs[action, obs]
        mem_belief = mem_belief @ mem_mat

    return mem_belief[mem]

if __name__ == "__main__":
    """
    For testing \hat{v}(o) = \sum_m p(m | h)\hat{v}(o, m)
    """
    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.2
    lambda_0 = 0.
    lambda_1 = 1.
    n_episode_samples = int(1)
    seed = 2022

    rand_key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    spec = load_spec(
        spec_name,
        # memory_id=str(mem),
        memory_id=str('fuzzy'),
        mem_leakiness=0.2,
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
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    traj_grad_fn = jax.grad(mem_traj_prob)

    mem_aug_amdp = memory_cross_product(mem_params, amdp)

    mem_probs = softmax(mem_params)

    n_mem_states = mem_params.shape[-1]

    print(f"Sampling {n_episode_samples} episodes")
    sampled_episodes = collect_episodes(amdp,
                                        pi,
                                        n_episode_samples,
                                        rand_key,
                                        mem_paramses=[mem_params, learnt_mem_params])

    lstd_v0, lstd_q0, lstd_info = lstdq_lambda(pi, amdp, lambda_=lambda_0)

    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi,
                                                           mem_aug_amdp,
                                                           lambda_=lambda_0)

    mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])

    init_mem_belief = jnp.zeros(n_mem_states)
    init_mem_belief = init_mem_belief.at[0].set(1.)
