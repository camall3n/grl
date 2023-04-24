from functools import partial
import jax
from jax.config import config
from jax.nn import softmax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from tqdm import tqdm

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
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
def expected_traj_grad(mem_grads: jnp.ndarray, mem_beliefs: jnp.ndarray, traj_vs: jnp.ndarray, t: int):
    """
    :param mem_grads: T x A x O x M x M
    :param mem_beliefs: T x A x O x M x M
    :param traj_vs: (T + 1)
    :param t: time step to consider
    """
    traj_grads_over_t = mem_grads[:t] * mem_beliefs[:t]
    traj_grads = traj_grads_over_t.sum(axis=0)

    return traj_vs[t] * traj_grads, traj_grads

@partial(jax.jit, static_argnames=['t', 'gamma'])
def expected_val_grad(mem_grads: jnp.ndarray, mem_beliefs: jnp.ndarray, traj_vs: jnp.ndarray, t: int, gamma: float):
    """
    :param mem_grads: T x A x O x M x M
    :param mem_beliefs: T x A x O x M x M
    :param traj_vs: (T + 1)
    :param t: time step to consider
    """
    total_timesteps = mem_grads.shape[0]
    discounts = gamma ** jnp.arange(1, total_timesteps - t)
    discounted_vs = discounts * traj_vs[t + 1:]
    value_weighted_mem_grads = jnp.einsum('i,ijklm->ijklm', discounted_vs, mem_grads[t:-1])
    val_grad = jnp.sum(value_weighted_mem_grads, axis=0)
    return mem_beliefs[t] * val_grad, val_grad

@partial(jax.jit, static_argnames='gamma')
def calc_episode_grads(episode: dict, init_mem_belief: jnp.ndarray,
                      mem_params: jnp.ndarray, learnt_mem_params: jnp.ndarray,
                      mem_lstd_v: jnp.ndarray, lstd_v1: jnp.ndarray,
                      mem_learnt_lstd_v: jnp.ndarray, gamma: float = 0.9):
    episode_length = episode['actions'].shape[0]
    oamm_counts = jnp.zeros_like(mem_params, dtype=int)
    oamm_learnt_counts = jnp.zeros_like(mem_params, dtype=int)

    mem_probs = softmax(mem_params, axis=-1)
    learnt_mem_probs = softmax(learnt_mem_params, axis=-1)

    # we need to batchify our mem_func
    batch_mem_grad_func = jax.vmap(jax.grad(mem_func), in_axes=(None, 0, 0, 0, 0))

    obses = episode['obses'][:-1]
    actions = episode['actions']
    prev_mem = episode['memses'][:-1]
    next_mem = episode['memses'][1:]

    # calculate mem gradients
    episode_mem_grads = batch_mem_grad_func(mem_params, obses, actions, prev_mem[:, 0], next_mem[:, 0])  # ep_length x |mems| x *mem_shape
    episode_learnt_mem_grads = batch_mem_grad_func(learnt_mem_params, obses, actions, prev_mem[:, 1], next_mem[:, 1])  # ep_length x |mems| x *mem_shape

    # Store our beliefs in a T x A x O x M x M tensor
    episode_mem_beliefs = jnp.zeros_like(episode_mem_grads)
    episode_learnt_mem_beliefs = jnp.zeros_like(episode_learnt_mem_grads)

    # initialize things we need to track
    mem_diffs, mem_beliefs = [], [init_mem_belief]
    learnt_mem_diffs, learnt_mem_beliefs = [], [init_mem_belief]

    mem_belief = init_mem_belief.copy()
    learnt_mem_belief = init_mem_belief.copy()

    traj_mem_vs = []
    traj_learnt_mem_vs = []

    for i, action in enumerate(episode['actions']):
        obs = episode['obses'][i]
        mem = episode['memses'][i][0]
        learnt_mem = episode['memses'][i][1]
        episode_mem_beliefs = episode_mem_beliefs.at[i, action, obs, mem].set(mem_belief)
        episode_learnt_mem_beliefs = episode_learnt_mem_beliefs.at[i, action, obs, mem].set(learnt_mem_belief)

        mem_mat = mem_probs[action, obs]
        learnt_mem_mat = learnt_mem_probs[action, obs]

        # update both our memory belief states
        mem_belief = mem_belief @ mem_mat
        learnt_mem_belief = learnt_mem_belief @ learnt_mem_mat

        mem_beliefs.append(mem_belief)
        learnt_mem_beliefs.append(learnt_mem_belief)

        mem_v_obs = mem_lstd_v[obs]
        learnt_mem_v_obs = mem_learnt_lstd_v[obs]

        traj_mem_vs.append(mem_v_obs[mem])
        traj_learnt_mem_vs.append(learnt_mem_v_obs[learnt_mem])

        # get our re-composed sample of our value function.
        mem_belief_weighted_v_obs = jnp.dot(mem_v_obs, mem_belief)
        learnt_mem_belief_weighted_v_obs = jnp.dot(learnt_mem_v_obs, learnt_mem_belief)

        target = lstd_v1[obs]
        mem_diff = mem_belief_weighted_v_obs - target
        learnt_mem_diff = learnt_mem_belief_weighted_v_obs - target

        mem_diffs.append(mem_diff)
        learnt_mem_diffs.append(learnt_mem_diff)

        oamm_counts = oamm_counts.at[action, obs, mem, episode['memses'][i + 1][0]].add(1)
        oamm_learnt_counts = oamm_learnt_counts.at[action, obs, learnt_mem, episode['memses'][i + 1][1]].add(1)

    mem_beliefs = jnp.array(mem_beliefs)
    learnt_mem_beliefs = jnp.array(learnt_mem_beliefs)
    traj_mem_vs = jnp.array(traj_mem_vs)
    traj_learnt_mem_vs = jnp.array(traj_learnt_mem_vs)
    mem_diffs = jnp.array(mem_diffs)
    learnt_mem_diffs = jnp.array(learnt_mem_diffs)

    # calculate grad statistics for mem
    traj_grad_mem_results = [expected_traj_grad(episode_mem_grads, episode_mem_beliefs, traj_mem_vs, t) for t in range(episode_length)]
    weighted_mem_traj_grads, mem_traj_grads = list(jnp.array(arr) for arr in zip(*traj_grad_mem_results))

    val_grad_mem_results = [expected_val_grad(episode_mem_grads, episode_mem_beliefs, traj_mem_vs, t, gamma) for t in range(episode_length)]
    weighted_mem_val_grads, mem_val_grads = list(jnp.array(arr) for arr in zip(*val_grad_mem_results))

    expected_episode_mem_grad = jnp.einsum('i,ijklm->jklm', mem_diffs, (weighted_mem_traj_grads + weighted_mem_val_grads))

    # now for our learnt mem
    traj_grad_learnt_mem_results = [expected_traj_grad(episode_learnt_mem_grads, episode_learnt_mem_beliefs, traj_learnt_mem_vs, t) for t in range(episode_length)]
    weighted_learnt_mem_traj_grads, learnt_mem_traj_grads = list(jnp.array(arr) for arr in zip(*traj_grad_learnt_mem_results))

    val_grad_learnt_mem_results = [expected_val_grad(episode_learnt_mem_grads, episode_learnt_mem_beliefs, traj_learnt_mem_vs, t, gamma) for t in range(episode_length)]
    weighted_learnt_mem_val_grads, learnt_mem_val_grads = list(jnp.array(arr) for arr in zip(*val_grad_learnt_mem_results))

    expected_episode_learnt_mem_grad = jnp.einsum('i,ijklm->jklm', learnt_mem_diffs, (weighted_learnt_mem_traj_grads + weighted_learnt_mem_val_grads))

    info = {
        'mem_traj_grads': mem_traj_grads,
        'mem_val_grads': mem_val_grads,
        'learnt_mem_traj_grads': learnt_mem_traj_grads,
        'learnt_mem_val_grads': learnt_mem_val_grads
    }

    return expected_episode_mem_grad, expected_episode_learnt_mem_grad, info

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
    n_episode_samples = int(1e4)
    seed = 2020


    rand_key = jax.random.PRNGKey(seed)

    # mem_funcs = [0, 16]
    # mem_funcs = [(19, 0), (16, 1)]
    agent_path = Path(ROOT_DIR, 'results', 'agents', 'tmaze_eps_hyperparams_seed(2026)_time(20230406-131003)_6a75e7a07d0b20088902a5094ede14cc.pkl.npy')
    learnt_mem_params, learnt_agent = load_mem_params(agent_path)

    # for mem, target_lambda in mem_funcs:

    spec = load_spec(spec_name,
                     # memory_id=str(mem),
                     memory_id=str(19),
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    pi = spec['Pi_phi'][0]
    mem_params = spec['mem_params']
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    grad_fn = jax.grad(obs_space_mem_discrep_loss)

    mem_aug_amdp = memory_cross_product(mem_params, amdp)
    learnt_mem_aug_amdp = memory_cross_product(learnt_mem_params, amdp)

    mem_probs = softmax(mem_params)
    learnt_mem_probs = softmax(learnt_mem_params)

    n_mem_states = mem_params.shape[-1]

    print(f"Sampling {n_episode_samples} episodes")
    sampled_episodes = collect_episodes(amdp, pi, n_episode_samples, rand_key, mem_paramses=[mem_params, learnt_mem_params])

    lstd_v0, lstd_q0, lstd_info = lstdq_lambda(pi, amdp, lambda_=lambda_0)
    lstd_v1, lstd_q1, lstd_info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

    mem_learnt_lstd_v0, mem_learnt_lstd_q0, mem_learnt_lstd_info = lstdq_lambda(mem_aug_pi, learnt_mem_aug_amdp)
    mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=lambda_0)
    mem_lstd_v1, mem_lstd_q1, mem_lstd_info_1 = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=lambda_1)

    mem_lstd_v0_unflat = mem_lstd_v0.reshape(-1, mem_params.shape[-1])
    mem_learnt_lstd_v0_unflat = mem_learnt_lstd_v0.reshape(-1, mem_params.shape[-1])

    init_mem_belief = np.zeros(n_mem_states)
    init_mem_belief[0] = 1.

    mem_sampled_v = jnp.zeros_like(lstd_v0)
    learnt_mem_sampled_v = jnp.zeros_like(mem_sampled_v)
    obs_counts = jnp.zeros_like(mem_sampled_v, dtype=int)

    expected_mem_grad = jnp.zeros_like(mem_params)
    expected_traj_mem_grad = jnp.zeros_like(mem_params)
    expected_val_mem_grad = jnp.zeros_like(mem_params)

    expected_learnt_mem_grad = jnp.zeros_like(learnt_mem_params)
    expected_traj_learnt_mem_grad = jnp.zeros_like(learnt_mem_params)
    expected_val_learnt_mem_grad = jnp.zeros_like(learnt_mem_params)

    for episode in tqdm(sampled_episodes):

        expected_episode_mem_grad, expected_episode_learnt_mem_grad, info \
            = calc_episode_grads(episode, init_mem_belief, mem_params, learnt_mem_params,
                                 mem_lstd_v0_unflat, lstd_v1, mem_learnt_lstd_v0_unflat,
                                 gamma=amdp.gamma)

        expected_mem_grad += expected_episode_mem_grad
        expected_learnt_mem_grad += expected_episode_learnt_mem_grad
        # mem_sampled_v += ep_mem_sampled_v
        # learnt_mem_sampled_v += ep_learnt_mem_sampled_v
        # obs_counts += ep_obs_counts

    # obs_counts = obs_counts.at[obs_counts == 0].add(1)
    # mem_sampled_v /= obs_counts
    # learnt_mem_sampled_v /= obs_counts

    analytical_mem_grad = grad_fn(mem_params, mem_aug_pi, amdp)
    analytical_learnt_mem_grad = grad_fn(learnt_mem_params, mem_aug_pi, amdp)

    expected_mem_grad /= len(sampled_episodes)
    expected_learnt_mem_grad /= len(sampled_episodes)

    diff_mem_grad = expected_mem_grad - analytical_mem_grad
    diff_learnt_mem_grad = expected_learnt_mem_grad - analytical_learnt_mem_grad
    print("done")
