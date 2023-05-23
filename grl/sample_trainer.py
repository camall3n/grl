from argparse import Namespace
from pathlib import Path
from typing import Union

from jax import random
import numpy as np
from tqdm import tqdm

from grl.mdp import MDP, AbstractMDP
from grl.agent.lstm import LSTMAgent
from grl.utils.data import Batch, one_hot
from grl.utils.replaymemory import EpisodeBuffer

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

        self.max_episode_steps = self.args.max_episode_steps
        self.trunc = self.args.trunc

        # For LSTMs
        self.action_cond = self.args.action_cond
        self.total_steps = self.args.total_steps

        self.checkpoint_freq = self.args.checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_all_checkpoints = self.args.save_all_checkpoints

        self.batch_size = self.args.batch_size

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
            episode_reward = []
            episode_loss = 0
            episode_info = {}

            checkpoint_after_ep = False

            network_params, optimizer_params, hs, self._rand_key = self.agent.reset(self._rand_key)

            # We do online training by resetting the buffer after every episode
            if self.online_training:
                self.buffer.reset()

            obs = self.env.reset()

            # Action conditioning for t=-1 action
            if self.action_cond == 'cat':
                action_encoding = np.zeros(self.env.n_actions)
                obs = np.concatenate([obs, action_encoding], axis=-1)

            action, self._rand_key, next_hs, qs = self.agent.act(network_params, obs, hs, self._rand_key).item()

            for t in range(self.max_episode_steps):
                next_obs, reward, done, info = self.env.step(action)

                # Action conditioning
                if self.action_cond == 'cat':
                    action_encoding = one_hot(action, self.env.n_actions)
                    next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                next_action, self._rand_key, next_next_hs = self.agent.act(network_params, next_obs, hs, self._rand_key).item()

                batch = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                              next_action=next_action, state=hs, next_state=next_hs,
                              end=done or (t == self.max_episode_steps - 1))
                self.buffer.push(batch)

                # if we have enough things in the batch
                if self.batch_size < len(self.buffer):

                    seq_len = self.agent.trunc
                    if self.agent.trunc <= 0:  # online training
                        seq_len = len(self.buffer)

                    # sample a sequence from our buffer!
                    sample = self.buffer.sample(self.batch_size, seq_len=seq_len)

                    sample.gamma = (1 - sample.done) * self.env.gamma

                    loss, network_params, optimizer_params = self.agent.update(network_params, optimizer_params, sample)

                pbar.update(1)
                self.num_steps += 1
                episode_reward.append(reward)

                # bump time step
                hs = next_hs
                next_hs = next_next_hs
                obs = next_obs
                action = next_action
