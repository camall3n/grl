import jax.numpy as jnp
from jax import random, jit
from jax.config import config
import numpy as np
from pathlib import Path
from tqdm import trange
from typing import Union

config.update('jax_platform_name', 'cpu')

from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import analytical_pe
from grl.utils.file_system import numpyify_and_save, load_info
from definitions import ROOT_DIR

@jit
def act(pi: jnp.ndarray, rand_key: random.PRNGKey):
    rand_key, choice_key = random.split(rand_key)
    action = random.choice(choice_key, pi.shape[-1], p=pi)
    return action, rand_key

def collect_episodes(mdp: Union[MDP, AbstractMDP],
                     pi: jnp.ndarray,
                     n_episodes: int,
                     rand_key: random.PRNGKey,
                     gamma_terminal: bool = False):
    episode_buffer = []

    for i in trange(n_episodes):
        obs, info = mdp.reset()
        done = False

        obses, actions, rewards, dones = [obs], [], [], []
        while not done:
            action, rand_key = act(pi[obs], rand_key)
            action = action.item()
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
        episode_buffer.append(episode)

    return episode_buffer


if __name__ == "__main__":
    spec_name = 'tmaze_5_two_thirds_up'
    n_step = float('inf')
    n_episodes = int(1e4)
    seed = 2023

    results_dir = Path(ROOT_DIR, 'scripts', 'results')
    buffer_path = results_dir / f'episode_buffer_spec({spec_name})_steps({n_step})_episodes({n_episodes:.0E})_seed({seed}).npy'

    rand_key = random.PRNGKey(seed)
    spec = load_spec(spec_name)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]

    analytical_state_vals, analytical_mc_vals, analytical_td_vals, _ = analytical_pe(pi, amdp)

    if buffer_path.is_file():
        episode_buffer = np.load(buffer_path, allow_pickle=True).tolist()
    else:
        episode_buffer = collect_episodes(amdp, pi, n_episodes, rand_key)
        numpyify_and_save(buffer_path, episode_buffer)

    n_step_sample_v = np.zeros(amdp.n_obs)
    n_step_sample_q = np.zeros((amdp.n_actions, amdp.n_obs))




