import jax
from jax import random
from jax.config import config
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from grl.environment import load_spec
from grl.memory import get_memory
from grl.mdp import MDP, POMDP, normalize
from grl.utils.mdp import get_td_model

def gather_counts(pomdp: POMDP,
                  pi: jnp.ndarray,
                  rand_key: random.PRNGKey,
                  n_samples: int = int(1e3)):
    pomdp_obs_count = np.zeros(pomdp.observation_space.n)

    obs, info = pomdp.reset()
    pomdp_obs_count[obs] += 1

    for i in trange(n_samples):

        # action_key, rand_key = random.split(rand_key)
        # a = random.choice(action_key, pomdp.action_space.n, p=pi[obs]).item()
        a = np.random.choice(pomdp.action_space.n, p=pi[obs])
        obs, reward, terminal, truncated, info = pomdp.step(a)
        pomdp_obs_count[obs] += 1
        if terminal:
            obs, info = pomdp.reset()
            pomdp_obs_count[obs] += 1

    return pomdp_obs_count

if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.2
    lambda_0 = 0.
    lambda_1 = 1.
    n_samples = int(1e6)
    seed = 2023

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
    pomdp = POMDP(mdp, spec['phi'])

    mem_params = get_memory('fuzzy',
                            n_obs=pomdp.observation_space.n,
                            n_actions=pomdp.action_space.n,
                            leakiness=0.2)

    pi = spec['Pi_phi'][0]

    T_obs_obs, R_obs_obs = get_td_model(pomdp, pi)
    T_obs_obs = normalize(T_obs_obs)
    td_mdp = MDP(T_obs_obs, R_obs_obs, pomdp.p0 @ pomdp.phi, gamma=pomdp.gamma)
    td_pomdp = POMDP(td_mdp, np.eye(td_mdp.state_space.n))

    print("Collecting POMDP samples")
    pomdp_obs_counts = gather_counts(pomdp, pi, rand_key, n_samples=n_samples)

    print("Collecting effective TD(0) model samples")
    pomdp_td_obs_counts = gather_counts(td_pomdp, pi, rand_key, n_samples=n_samples)

    pomdp_obs_counts /= n_samples
    pomdp_td_obs_counts /= n_samples
    print("collected")
