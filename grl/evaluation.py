from typing import Union

from jax import random
from tqdm import trange

from grl.agent.lstm import LSTMAgent
from grl.mdp import MDP, AbstractMDP

def test_episodes(agent: LSTMAgent, network_params: dict,
                  env: Union[MDP, AbstractMDP], rand_key: random.PRNGKey,
                  n_episodes: int = 1, test_eps: float = 0.):
    rews = []
    original_epsilon = agent.eps
    agent.eps = test_eps

    for ep in trange(n_episodes):
        ep_rews = []

        prev_hs, rand_key = agent.reset(rand_key)

        obs, env_info = env.reset()


    agent.eps = original_epsilon
