from argparse import Namespace
from pathlib import Path
from typing import Union

from jax import random
from tqdm import tqdm

from grl.mdp import MDP, AbstractMDP
from grl.agent.lstm import LSTMAgent
from grl.utils.replaymemory import ReplayBuffer, EpisodeBuffer

class Trainer:
    def __init__(self,
                 env: Union[MDP, AbstractMDP],
                 agent: LSTMAgent,
                 rand_key: random.PRNGKey,
                 args: Namespace,
                 checkpoint_dir: Path = None):
        self.env = env
        self.agent = agent
        self.args = args
        self._rand_key = rand_key

        self.max_episode_steps = args.max_episode_steps

        # For LSTMs
        self.action_cond = args.action_cond
        self.total_steps = args.total_steps

        self.checkpoint_freq = args.checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_all_checkpoints = args.save_all_checkpoints

        self.batch_size = args.batch_size

        # if 'lstm' in self.args.arch and isinstance(self.agent, LSTMAgent):
        self.online_training = self.args.buffer_size <= 1

        replay_capacity = self.args.buffer_size
        if replay_capacity < self.max_episode_steps:
            replay_capacity = self.max_episode_steps

        # We save state for an LSTM agent
        obs_shape = self.env.observation_space.shape
        if self.action_cond == 'cat':
            obs_shape = obs_shape[:-1] + (obs_shape[-1] + self.env.n_actions,)

        self.buffer = EpisodeBuffer(replay_capacity, rand_key, obs_shape,
                                    obs_dtype=self.env.observation_space.low.dtype,
                                    state_size=(self.args.hidden_size, ))
        # else:
        #     self.buffer = ReplayBuffer(args.buffer_size, rand_key, self.env.observation_space.shape,
        #                                obs_dtype=self.env.observation_space.low.dtype)

        self.episode_num = 0
        self.num_steps = 0

    def train(self, log_every: int = 1000):

        pbar = tqdm(total=self.total_steps)

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0
            episode_info = {}

            checkpoint_after_ep = False

            network_params, optimizer_params, hs, self._rand_key = self.agent.reset(self._rand_key)

            # LSTM hidden state
            next_hs = None

            obs = self.env.reset()

            action = self.agent.act(obs).item()

            for t in range(self.max_episode_steps):
                next_obs, reward, done, info = self.env.step(action)

                pbar.update(1)
