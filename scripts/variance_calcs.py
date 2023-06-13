from functools import partial
import jax.numpy as jnp
from jax import random, jit, nn
from jax.config import config
import numpy as np
from pathlib import Path
from tqdm import trange, tqdm
from typing import Union, List, Tuple

config.update('jax_platform_name', 'cpu')

from grl.environment import load_spec
from grl.mdp import MDP, POMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.file_system import numpyify_and_save, load_info
from definitions import ROOT_DIR

@jit
def act(pi: jnp.ndarray, rand_key: random.PRNGKey):
    rand_key, choice_key = random.split(rand_key)
    action = random.choice(choice_key, pi.shape[-1], p=pi)
    return action, rand_key

@partial(jit, static_argnames=['prev_mems', 'action', 'obs'])
def mem_act(mem_probses: jnp.ndarray, prev_mems: Tuple[int], action: int, obs: int,
            rand_key: random.PRNGKey):
    all_keys = random.split(rand_key, num=len(mem_probses) + 1)
    mem_keys = all_keys[:-1]
    rand_key = all_keys[-1]
    return tuple(
        random.choice(mem_key, mem_prob.shape[-1], p=mem_prob[action, obs, prev_mems[i]])
        for i, (mem_key, mem_prob) in enumerate(zip(mem_keys, mem_probses))), rand_key

def collect_episodes(mdp: Union[MDP, POMDP],
                     pi: jnp.ndarray,
                     n_episodes: int,
                     rand_key: random.PRNGKey,
                     gamma_terminal: bool = False,
                     mem_paramses: List[jnp.ndarray] = None):
    episode_buffer = []
    mem_probses = None
    if mem_paramses is not None:
        mem_probses = [nn.softmax(mem_params, axis=-1) for mem_params in mem_paramses]

    for i in trange(n_episodes):
        obs, info = mdp.reset()
        mem, mems = None, None

        if mem_paramses is not None:
            mem = tuple([0] * len(mem_paramses))
            mems = [mem]

        done = False

        obses, actions, rewards, dones = [obs], [], [], []
        while not done:
            action, rand_key = act(pi[obs], rand_key)
            action = action.item()

            if mem_paramses is not None:
                mem, rand_key = mem_act(mem_probses, mem, action, obs, rand_key)
                mem = tuple(m.item() for m in mem)
                mems.append(mem)

            obs, reward, done, truncated, info = mdp.step(action, gamma_terminal=gamma_terminal)

            obses.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        episode = {
            'obses': np.array(obses, dtype=np.uint8),
            'actions': np.array(actions, dtype=np.uint8),
            'rewards': np.array(rewards, dtype=float),
            'dones': np.array(dones, dtype=bool)
        }
        if mem_paramses is not None:
            episode['memses'] = np.array(mems, dtype=np.uint8)
        episode_buffer.append(episode)

    return episode_buffer

def rews_to_returns(rews: np.ndarray, gamma: float):
    returns = np.zeros_like(rews)

    returns[-1] = rews[-1]
    remaining_rews = rews[::-1][1:]
    for i, rew in enumerate(remaining_rews, start=1):
        returns[len(remaining_rews) - i] = rew + gamma * returns[len(rews) - i]
    return returns

def get_act_obs_idx(actions: np.ndarray, obs: np.ndarray, n_obs: int):
    return actions * n_obs + obs

if __name__ == "__main__":
    spec_name = "tmaze_5_two_thirds_up_fully_observable"
    spec_hparams = None
    # spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 2 / 3
    epsilon = 0
    # spec_hparams = (corridor_length, discount, junction_up_pi, epsilon)

    n_step = float('inf')
    n_episodes = int(1e4)
    lambda_0 = 0.
    lambda_1 = 1.
    seed = 2023

    results_dir = Path(ROOT_DIR, 'scripts', 'results')
    buffer_path = results_dir / f'episode_buffer_spec({spec_name}_{str(spec_hparams)})_steps({n_step})_episodes({n_episodes:.0E})_seed({seed}).npy'

    rand_key = random.PRNGKey(seed)
    spec = load_spec(spec_name,
                     memory_id=str(16),
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]
    n_obs = amdp.observation_space.n
    n_actions = amdp.action_space.n
    n_oa = n_obs * n_actions

    analytical_v0_vals, analytical_q0_vals, info_0 = lstdq_lambda(pi, amdp, lambda_=lambda_0)
    analytical_v1_vals, analytical_q1_vals, info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

    if buffer_path.is_file():
        episode_buffer = np.load(buffer_path, allow_pickle=True).tolist()
    else:
        episode_buffer = collect_episodes(amdp, pi, n_episodes, rand_key)
        numpyify_and_save(buffer_path, episode_buffer)

    sample_v = np.zeros(amdp.observation_space.n)
    sample_q = np.zeros(n_oa)
    all_act_obs_counts = np.zeros_like(sample_q, dtype=int)

    var_q = np.zeros_like(all_act_obs_counts, dtype=float)
    mse_q = np.zeros_like(all_act_obs_counts, dtype=float)

    for ep in tqdm(episode_buffer):
        rewards = ep['rewards']
        returns = rews_to_returns(rewards, amdp.gamma)
        actions = ep['actions']
        obs = ep['obses'][:returns.shape[0]]
        next_obs = ep['obses'][1:]

        act_obs_idxes = get_act_obs_idx(actions, obs, n_obs=n_obs)
        all_act_obs_counts += np.bincount(act_obs_idxes, minlength=all_act_obs_counts.shape[0])

        sample_q += np.bincount(act_obs_idxes, weights=returns, minlength=n_oa)

        one_step_returns = rewards + amdp.gamma * analytical_v0_vals[next_obs]
        vars = (one_step_returns - analytical_q0_vals[actions, obs])**2
        var_q += np.bincount(act_obs_idxes, weights=vars, minlength=n_oa)

        mse = (one_step_returns - returns)**2
        mse_q += np.bincount(act_obs_idxes, weights=mse, minlength=n_oa)

    denom = all_act_obs_counts + (all_act_obs_counts == 0)
    mse_q /= denom
    var_q /= denom
    bias_q = np.array((analytical_q1_vals - analytical_q0_vals)**2)

    mse_q = mse_q.reshape((n_actions, n_obs))
    var_q = var_q.reshape((n_actions, n_obs))
    print(f"returns: {returns}")
