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
from grl.utils.loss import obs_space_mem_discrep_loss
from grl.utils.policy_eval import lstdq_lambda

from scripts.variance_calcs import collect_episodes
from definitions import ROOT_DIR

def load_mem_params(agent_path: Path):
    agent = np.load(agent_path, allow_pickle=True).item()
    return agent.mem_params, agent

def mem_func(mem_params: jnp.ndarray, obs: int, action: int, prev_mem: int, next_mem: int):
    mem_probs = softmax(mem_params, axis=-1)
    return mem_probs[action, obs, prev_mem, next_mem]

@partial(jax.jit, static_argnames='t')
def expected_val_grad(mem_grads: jnp.ndarray, mem_beliefs: jnp.ndarray, traj_vs: jnp.ndarray,
                      discounts: jnp.ndarray, t: int):
    """
    :param mem_grads: T x A x O x M x M
    :param mem_beliefs: T x A x O x M x M
    :param traj_vs: (T + 1)
    :param t: time step to consider
    """
    T = discounts.shape[0]
    discounted_vs = discounts[1:T - t + 1] * traj_vs[t:]
    value_weighted_mem_grads = jnp.einsum('i,ijklm->ijklm', discounted_vs, mem_grads[t - 1:])
    val_grad = jnp.sum(value_weighted_mem_grads, axis=0)
    return mem_beliefs[t] * val_grad, val_grad

@partial(jax.jit, static_argnames=['gamma', 'mem_idx'])
def calc_episode_grads(episode: dict,
                       init_mem_belief: jnp.ndarray,
                       mem_params: jnp.ndarray,
                       mem_lstd_v: jnp.ndarray,
                       lstd_v1: jnp.ndarray,
                       gamma: float = 0.9,
                       mem_idx: int = 0):
    T = episode['obses'].shape[0]
    oamm_counts = jnp.zeros_like(mem_params, dtype=int)

    mem_probs = softmax(mem_params, axis=-1)

    # we need to batchify our mem_func
    batch_mem_grad_func = jax.vmap(jax.grad(mem_func), in_axes=(None, 0, 0, 0, 0))

    obses = episode['obses'][:-1]
    actions = episode['actions']
    prev_mem = episode['memses'][:-1]
    next_mem = episode['memses'][1:]

    # calculate mem gradients
    episode_mem_grads = batch_mem_grad_func(mem_params, obses, actions, prev_mem[:, mem_idx],
                                            next_mem[:, mem_idx]) # ep_length x |mems| x *mem_shape

    # store mem beliefs over this (o, m) episode
    episode_mem_beliefs = []
    discounts = []

    # initialize things we need to track
    mem_diffs, mem_beliefs = [], []
    mem_belief = init_mem_belief.copy()

    traj_mem_vs = []

    # we add the -1 here because there are T - 1 actions in a trajectory.
    for t in range(T - 1):
        action = episode['actions'][t]
        obs = episode['obses'][t]
        mem = episode['memses'][t][mem_idx]

        # add our belief states and beliefs to buffers
        mem_beliefs.append(mem_belief)
        episode_mem_beliefs.append(mem_belief[mem])

        # here we calculate our values
        mem_v_obs = mem_lstd_v[obs]

        traj_mem_vs.append(mem_v_obs[mem])

        # get our re-composed sample of our value function.
        mem_belief_weighted_v_obs = jnp.dot(mem_v_obs, mem_belief)

        target = lstd_v1[obs]
        mem_diff = mem_belief_weighted_v_obs - target

        mem_diffs.append(mem_diff)
        discounts.append(t)

        oamm_counts = oamm_counts.at[action, obs, mem, episode['memses'][t + 1][0]].add(1)

        # update both our memory belief states for t + 1
        mem_mat = mem_probs[action, obs]
        mem_belief = mem_belief @ mem_mat

    else:
        obs = episode['obses'][t + 1]
        mem = episode['memses'][t + 1][mem_idx]

        # update finals beliefs
        mem_beliefs.append(mem_belief)
        episode_mem_beliefs.append(mem_belief[mem])

        mem_v_obs = mem_lstd_v[obs]

        traj_mem_vs.append(mem_v_obs[mem])

        # get our re-composed sample of our value function.
        mem_belief_weighted_v_obs = jnp.dot(mem_v_obs, mem_belief)

        target = lstd_v1[obs]
        mem_diff = mem_belief_weighted_v_obs - target

        mem_diffs.append(mem_diff)
        discounts.append(t + 1)

    mem_beliefs = jnp.array(mem_beliefs)
    episode_mem_beliefs = jnp.array(episode_mem_beliefs)
    traj_mem_vs = jnp.array(traj_mem_vs)
    mem_diffs = jnp.array(mem_diffs)

    discounts = amdp.gamma**jnp.arange(T)
    discounts = discounts.at[-1].set(0)

    # calculate grad statistics for mem
    traj_grad_mem_results = [
        expected_traj_grad(episode_mem_grads, episode_mem_beliefs, traj_mem_vs, t)
        for t in range(1, T)
    ]
    weighted_mem_traj_grads, mem_traj_grads = list(
        jnp.array(arr) for arr in zip(*traj_grad_mem_results))

    val_grad_mem_results = [
        expected_val_grad(episode_mem_grads, episode_mem_beliefs, traj_mem_vs, discounts, t)
        for t in range(1, T)
    ]
    weighted_mem_val_grads, mem_val_grads = list(
        jnp.array(arr) for arr in zip(*val_grad_mem_results))

    expected_episode_mem_grad = jnp.einsum('i,ijklm->jklm', mem_diffs,
                                           (weighted_mem_traj_grads + weighted_mem_val_grads))

    info = {
        'mem_traj_grads': mem_traj_grads,
        'mem_val_grads': mem_val_grads,
    }

    return expected_episode_mem_grad, info

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
    seed = 2023

    rand_key = jax.random.PRNGKey(seed)

    # mem_funcs = [0, 16]
    # mem_funcs = [(19, 0), (16, 1)]
    agent_path = Path(
        ROOT_DIR, 'results', 'agent',
        'tmaze_eps_hyperparams_seed(2026)_time(20230406-131003)_6a75e7a07d0b20088902a5094ede14cc.pkl.npy'
    )
    learnt_mem_params, learnt_agent = load_mem_params(agent_path)

    # for mem, target_lambda in mem_funcs:

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

    grad_fn = jax.grad(obs_space_mem_discrep_loss)

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
    lstd_v1, lstd_q1, lstd_info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

    mem_learnt_lstd_v0, mem_learnt_lstd_q0, mem_learnt_lstd_info = lstdq_lambda(
        mem_aug_pi, learnt_mem_aug_amdp)
    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi,
                                                           mem_aug_amdp,
                                                           lambda_=lambda_0)
    mem_lstd_v1, mem_lstd_q1, mem_lstd_info_1 = lstdq_lambda(mem_aug_pi,
                                                             mem_aug_amdp,
                                                             lambda_=lambda_1)

    mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])
    learnt_mem_lstd_v0_unflat = mem_learnt_lstd_v0.reshape(-1, mem_params.shape[-1])

    init_mem_belief = np.zeros(n_mem_states)
    init_mem_belief[0] = 1.

    mem_sampled_v = jnp.zeros_like(lstd_v0)
    learnt_mem_sampled_v = jnp.zeros_like(mem_sampled_v)
    obs_counts = jnp.zeros_like(mem_sampled_v, dtype=int)

    expected_mem_grad = jnp.zeros_like(mem_params)

    expected_learnt_mem_grad = jnp.zeros_like(learnt_mem_params)

    for episode in tqdm(sampled_episodes):

        expected_episode_mem_grad, mem_info \
            = calc_episode_grads(episode, init_mem_belief, mem_params,
                                 mem_lstd_v0_unflat, lstd_v1,
                                 gamma=amdp.gamma, mem_idx=0)

        expected_mem_grad += expected_episode_mem_grad

        expected_episode_learnt_mem_grad, learnt_mem_info \
            = calc_episode_grads(episode, init_mem_belief, learnt_mem_params,
                                 learnt_mem_lstd_v0_unflat, lstd_v1,
                                 gamma=amdp.gamma, mem_idx=1)
        expected_learnt_mem_grad += expected_episode_learnt_mem_grad

    analytical_mem_grad = grad_fn(mem_params, mem_aug_pi, amdp)
    analytical_learnt_mem_grad = grad_fn(learnt_mem_params, mem_aug_pi, amdp)

    expected_mem_grad /= len(sampled_episodes)
    expected_learnt_mem_grad /= len(sampled_episodes)

    avg_analytical_mem_grad = analytical_mem_grad.mean()
    avg_analytical_learnt_mem_grad = analytical_learnt_mem_grad.mean()

    diff_mem_grad = expected_mem_grad - analytical_mem_grad
    diff_learnt_mem_grad = expected_learnt_mem_grad - analytical_learnt_mem_grad
    print("done")
