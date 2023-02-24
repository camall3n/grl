import jax.numpy as jnp
from jax import random, jit, lax
from jax.config import config
import numpy as np
import pytest
from tqdm import trange
from functools import partial

config.update('jax_platform_name', 'cpu')
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.memory import memory_cross_product
from grl.utils.mdp import functional_get_occupancy

@jit
def act(pi: jnp.ndarray, rand_key: random.PRNGKey):
    rand_key, choice_key = random.split(rand_key)
    action = random.choice(choice_key, pi.shape[-1], p=pi)
    return action, rand_key

@partial(jit, static_argnames=['mdp', 'current_state', 'action'])
def step(mdp: MDP, current_state: int, action: int, rand_key: random.PRNGKey):
    pr_next_s = mdp.T[action, current_state, :]
    rand_key, choice_key, terminal_key = random.split(rand_key, 3)
    next_state = random.choice(choice_key, mdp.n_states, p=pr_next_s)
    reward = mdp.R[action][current_state][next_state]

    # Check if next_state is absorbing state
    is_absorbing = (mdp.T[:, next_state, next_state] == 1)

    # Discounting: end episode with probability 1-gamma
    terminal = is_absorbing.all() | (random.uniform(terminal_key) < (1 - mdp.gamma)) # absorbing for all actions

    truncated = False
    observation = next_state
    info = {'state': next_state, 'rand_key': rand_key}
    # Conform to new-style Gym API
    return observation, reward, terminal, truncated, info

# @pytest.mark.slow
def test_count():
    seed = 2020
    samples = int(1e5)
    spec_name = 'tmaze_eps_hyperparams'

    rand_key = random.PRNGKey(seed)
    spec = load_spec(spec_name,
                     memory_id=str(0),
                     corridor_length=5,
                     discount=0.9,
                     junction_up_pi=2/3,
                     epsilon=0.1)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    n_mem_states = spec['mem_params'].shape[-1]
    # def cumulative_bitwise_and(carry: Tuple, )

    # id_mask = jnp.expand_dims(jnp.eye(amdp.T.shape[-1]), 0).repeat(amdp.T.shape[0], axis=0)
    # self_loop = (id_mask * amdp.T).astype(bool)
    # absorbing_mask = lax.reduce(self_loop, jnp.array(1).astype(bool), jnp.bitwise_and, (0,))
    mem_aug_amdp = memory_cross_product(spec['mem_params'], amdp)

    # Make the last two memory-augmented states absorbing as well
    mem_aug_amdp.T = mem_aug_amdp.T.at[:, -2:].set(0)
    mem_aug_amdp.T = mem_aug_amdp.T.at[:, -2, -2].set(1)
    mem_aug_amdp.T = mem_aug_amdp.T.at[:, -1, -1].set(1)

    mem_aug_mdp = MDP(mem_aug_amdp.T, mem_aug_amdp.R, mem_aug_amdp.p0, mem_aug_amdp.gamma)
    pi_ground = (spec['phi'] @ spec['Pi_phi'][0]).repeat(n_mem_states, 0)

    state_counts = np.zeros(mem_aug_mdp.n_states, dtype=int)
    print(f"Collecting {samples} samples from {spec_name} spec")
    obs, info = mem_aug_mdp.reset()
    obs = jnp.array(obs)

    for i in trange(samples):
        state_counts[obs] += 1

        action, rand_key = act(pi_ground[obs], rand_key)

        obs, reward, terminal, truncated, info = step(mem_aug_mdp, obs.item(), action.item(), rand_key)
        rand_key = info['rand_key']

        if terminal:
            obs, info = mem_aug_mdp.reset()
            obs = jnp.array(obs)


    sampled_count_dist = state_counts / state_counts.sum(axis=-1, keepdims=True)
    print("Done collecting samples.")

    print("Calculating analytical occupancy")
    c_s = functional_get_occupancy(pi_ground, mem_aug_mdp)
    c_s = c_s.at[-2:].set(0)
    analytical_count_dist = c_s / c_s.sum(axis=-1, keepdims=True)

    assert jnp.allclose(sampled_count_dist, analytical_count_dist, atol=1e-3)







if __name__ == "__main__":
    test_count()
