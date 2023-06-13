from functools import partial
import jax
from jax.config import config
from jax.nn import softmax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from tqdm import tqdm

from grl.environment import load_spec
from grl.memory import memory_cross_product, get_memory
from grl.mdp import MDP, POMDP
from grl.utils.policy_eval import lstdq_lambda

from definitions import ROOT_DIR
from scripts.variance_calcs import collect_episodes
from scripts.memory_gradient.intermediate_sample_grads import expected_val_grad, mem_func, load_mem_params

def mem_obs_val_func(mem_params: jnp.ndarray, amdp: POMDP, pi: jnp.ndarray, obs: int, mem: int):

    # T_td, R_td = get_td_model(amdp, pi)
    # td_model = MDP(T_td, R_td, amdp.p0 @ amdp.phi, gamma=amdp.gamma)
    # v0, q0 = functional_solve_mdp(pi, td_model)
    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    v0, q0, info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=0)
    v0_unflat = v0.reshape(-1, mem_params.shape[-1])
    return v0_unflat[obs, mem]

@partial(jax.jit, static_argnames=['gamma', 'mem_idx'])
def seq_calc_episode_val_grads(episode: dict,
                               init_mem_belief: jnp.ndarray,
                               mem_params: jnp.ndarray,
                               mem_lstd_v: jnp.ndarray,
                               mem_idx: int = 0):
    T = episode['obses'].shape[0]
    n_obs = mem_params.shape[1]
    n_mem = mem_params.shape[-1]

    obses = episode['obses']
    memses = episode['memses'][:, mem_idx]
    actions = episode['actions']
    prev_mem = episode['memses'][:-1, mem_idx]
    next_mem = episode['memses'][1:, mem_idx]

    batch_mem_grad_func = jax.vmap(jax.grad(mem_func), in_axes=(None, 0, 0, 0, 0))

    # calculate mem gradients, indices are from t = (0, 1) ... (T - 1, T)
    episode_mem_grads = batch_mem_grad_func(mem_params, obses[:-1], actions, prev_mem,
                                            next_mem) # ep_length x |mems| x *mem_shape

    discounts = amdp.gamma**jnp.arange(T)
    discounts = discounts.at[-1].set(0)

    om_counts = jnp.zeros((n_obs, n_mem))
    val_grad_buffer = jnp.zeros((n_obs, n_mem) + mem_params.shape)

    # t is i in eq. 10
    for t in range(1, T - 1):
        action = episode['actions'][t]
        obs = episode['obses'][t]
        mem = episode['memses'][t][mem_idx]
        om_counts = om_counts.at[obs, mem].add(1)

        t_discounts = discounts[1:T - t + 1]
        t_vals = mem_lstd_v[episode['obses'][t:], episode['memses'][t:, mem_idx]]
        t_discounted_vals = t_vals * t_discounts
        t_mem_grads = episode_mem_grads[t - 1:]

        t_grads = jnp.einsum('i,ijklm->jklm', t_discounted_vals, t_mem_grads)
        val_grad_buffer = val_grad_buffer.at[obs, mem].add(t_grads)

    om_counts += (om_counts == 0)
    # val_grad_buffer = jnp.einsum('ij,ijklmn->ijklmn', 1/om_counts, val_grad_buffer)
    return val_grad_buffer

@partial(jax.jit, static_argnames=['gamma', 'mem_idx'])
def calc_episode_val_grads(episode: dict,
                           init_mem_belief: jnp.ndarray,
                           mem_params: jnp.ndarray,
                           mem_lstd_v: jnp.ndarray,
                           mem_idx: int = 0):
    T = episode['obses'].shape[0]
    n_obs = mem_params.shape[1]
    n_mem = mem_params.shape[-1]
    om_counts = jnp.zeros((n_obs, n_mem))

    mem_probs = softmax(mem_params, axis=-1)

    # we need to batchify our mem_func
    batch_mem_grad_func = jax.vmap(jax.grad(mem_func), in_axes=(None, 0, 0, 0, 0))

    obses = episode['obses']
    memses = episode['memses'][:, mem_idx]
    actions = episode['actions']
    prev_mem = episode['memses'][:-1, mem_idx]
    next_mem = episode['memses'][1:, mem_idx]

    # calculate mem gradients, indices are from t = (0, 1) ... (T - 1, T)
    episode_mem_grads = batch_mem_grad_func(mem_params, obses[:-1], actions, prev_mem,
                                            next_mem) # ep_length x |mems| x *mem_shape

    # store mem beliefs over this (o, m) episode
    episode_mem_beliefs = []

    # initialize things we need to track
    mem_diffs, mem_beliefs = [], []
    mem_belief = init_mem_belief.copy()

    traj_mem_vs = []

    # we add the -1 here because there are T - 1 actions in a trajectory.
    for t in range(T - 1):
        action = episode['actions'][t]
        obs = episode['obses'][t]
        mem = episode['memses'][t][mem_idx]
        om_counts = om_counts.at[obs, mem].add(1)

        # add our belief states and beliefs to buffers
        mem_beliefs.append(mem_belief)
        episode_mem_beliefs.append(mem_belief[mem])

        # here we calculate our values
        mem_v_obs = mem_lstd_v[obs]

        traj_mem_vs.append(mem_v_obs[mem])

        # update both our memory belief states for t + 1
        mem_mat = mem_probs[action, obs]
        mem_belief = mem_belief @ mem_mat

    else:
        obs = episode['obses'][t + 1]
        mem = episode['memses'][t + 1][mem_idx]
        om_counts = om_counts.at[obs, mem].add(1)

        # update finals beliefs
        mem_beliefs.append(mem_belief)
        episode_mem_beliefs.append(mem_belief[mem])

        mem_v_obs = mem_lstd_v[obs]

        traj_mem_vs.append(mem_v_obs[mem])

    mem_beliefs = jnp.array(mem_beliefs)
    episode_mem_beliefs = jnp.array(episode_mem_beliefs)
    traj_mem_vs = jnp.array(traj_mem_vs)

    discounts = amdp.gamma**jnp.arange(T)
    discounts = discounts.at[-1].set(0)

    # calculate grad statistics for mem
    val_grad_buffer = jnp.zeros((n_obs, n_mem) + mem_params.shape)
    for t in range(1, T):
        weighted_val_grad_mem, val_grad_mem = \
            expected_val_grad(episode_mem_grads, episode_mem_beliefs, traj_mem_vs, discounts, t)
        val_grad_buffer = val_grad_buffer.at[episode['obses'][t],
                                             episode['memses'][t]].add(val_grad_mem)

    # Normalize over obs
    om_counts += (om_counts == 0)
    val_grad_buffer = jnp.einsum('ij,ijklmn->ijklmn', 1 / om_counts, val_grad_buffer)

    seq_val_grad = seq_calc_episode_val_grads(episode,
                                              init_mem_belief,
                                              mem_params,
                                              mem_lstd_v,
                                              mem_idx=mem_idx)
    return val_grad_buffer

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
    n_episode_samples = int(10000)
    seed = 2022

    rand_key = jax.random.PRNGKey(seed)

    agent_path = Path(
        ROOT_DIR, 'results', 'agent',
        'tmaze_eps_hyperparams_seed(2026)_time(20230406-131003)_6a75e7a07d0b20088902a5094ede14cc.pkl.npy'
    )
    learnt_mem_params, learnt_agent = load_mem_params(agent_path)

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
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    grad_fn = jax.grad(mem_obs_val_func)

    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    learnt_mem_aug_amdp = memory_cross_product(learnt_mem_params, amdp)

    mem_probs = softmax(mem_params)
    learnt_mem_probs = softmax(learnt_mem_params)

    n_mem_states = mem_params.shape[-1]

    print(f"Sampling {n_episode_samples} episodes")
    sampled_episodes = collect_episodes(amdp,
                                        pi,
                                        n_episode_samples,
                                        rand_key,
                                        mem_paramses=[mem_params, learnt_mem_params])

    lstd_v0, lstd_q0, lstd_info = lstdq_lambda(pi, amdp, lambda_=lambda_0)

    mem_learnt_lstd_v0, mem_learnt_lstd_q0, mem_learnt_lstd_info = lstdq_lambda(
        mem_aug_pi, learnt_mem_aug_amdp)
    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi,
                                                           mem_aug_amdp,
                                                           lambda_=lambda_0)

    mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])
    learnt_mem_lstd_v0_unflat = mem_learnt_lstd_v0.reshape(-1, mem_params.shape[-1])

    init_mem_belief = np.zeros(n_mem_states)
    init_mem_belief[0] = 1.

    expected_mem_val_grads = jnp.zeros((amdp.observation_space.n, n_mem_states) + mem_params.shape)
    expected_learnt_mem_val_grads = jnp.zeros((amdp.observation_space.n, n_mem_states) +
                                              mem_params.shape)

    analytical_mem_val_grad = jnp.zeros_like(expected_mem_val_grads)
    analytical_learnt_mem_val_grad = jnp.zeros_like(expected_learnt_mem_val_grads)

    for o in range(amdp.phi.shape[-1]):
        for m in range(n_mem_states):
            analytical_mem_val_grad = analytical_mem_val_grad.at[o, m].set(
                grad_fn(mem_params, amdp, pi, o, m))
            analytical_learnt_mem_val_grad = analytical_learnt_mem_val_grad.at[o, m].set(
                grad_fn(learnt_mem_params, amdp, pi, o, m))

    for episode in tqdm(sampled_episodes):

        # with jax.disable_jit():
        episode_mem_val_grads \
            = seq_calc_episode_val_grads(episode, init_mem_belief, mem_params,
                                     mem_lstd_v0_unflat, mem_idx=0)

        expected_mem_val_grads += episode_mem_val_grads

        # episode_learnt_mem_val_grads \
        #     = calc_episode_val_grads(episode, init_mem_belief, learnt_mem_params,
        #                              learnt_mem_lstd_v0_unflat, mem_idx=1)
        # expected_learnt_mem_val_grads += episode_learnt_mem_val_grads

    expected_mem_val_grads /= len(sampled_episodes)
    expected_learnt_mem_val_grads /= len(sampled_episodes)

    # avg_analytical_mem_grad = analytical_mem_grad.mean()
    # avg_analytical_learnt_mem_grad = analytical_learnt_mem_grad.mean()
    #
    # diff_mem_grad = expected_mem_grad - analytical_mem_grad
    # diff_learnt_mem_grad = expected_learnt_mem_grad - analytical_learnt_mem_grad
    print("done")
