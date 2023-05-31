from argparse import Namespace
from pathlib import Path
from typing import Union, List, Tuple

from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint
from tqdm import tqdm

from grl.evaluation import test_episodes
from grl.mdp import MDP, AbstractMDP
from grl.agent.rnn import RNNAgent
from grl.utils.data import Batch, one_hot, compress_episode_rewards
from grl.utils.replaymemory import EpisodeBuffer

class Trainer:
    def __init__(self,
                 env: Union[MDP, AbstractMDP],
                 agent: RNNAgent,
                 rand_key: random.PRNGKey,
                 args: Namespace,
                 checkpoint_dir: Path = None):
        self.env = env
        self.agent = agent
        self.args = args
        self._rand_key = rand_key

        self.gamma_terminal = not self.args.no_gamma_terminal
        self.discounting = self.env.gamma if not self.gamma_terminal else 1.

        self.max_episode_steps = self.args.max_episode_steps
        self.trunc = self.args.trunc

        # For RNNs
        self.action_cond = self.args.action_cond
        self.total_steps = self.args.total_steps

        # we train at the end of an episode only if we also need to learn an MC head.
        self.include_returns_in_batch = self.args.algo == 'multihead_rnn' and \
                                        self.args.multihead_loss_mode in ['both', 'mc', 'split']

        # Logging and eval
        self.offline_eval_freq = self.args.offline_eval_freq
        self.offline_eval_episodes = self.args.offline_eval_episodes
        self.offline_eval_epsilon = self.args.offline_eval_epsilon
        self.checkpoint_freq = self.args.checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_all_checkpoints = self.args.save_all_checkpoints

        self.batch_size = self.args.batch_size
        self.online_training = self.args.replay_size <= 1

        # Replay buffer initialization
        replay_capacity = self.args.replay_size
        if replay_capacity < self.max_episode_steps:
            replay_capacity = self.max_episode_steps

        # TODO: remove all of this! Refactor MDPs/AbstractMDPs into a environment-looking thing.
        obs_shape = self.env.observation_space
        if self.action_cond == 'cat':
            obs_shape = obs_shape[:-1] + (obs_shape[-1] + self.env.n_actions,)

        self.one_hot_obses = isinstance(self.env, AbstractMDP) or isinstance(self.env, MDP)

        state_size = (self.args.hidden_size, )
        if self.args.arch == 'lstm':
            state_size = (2, ) + state_size

        self.buffer = EpisodeBuffer(replay_capacity, rand_key, obs_shape,
                                    state_size=state_size, unpack_state=self.args.arch == 'lstm')

        # Initialize checkpointers
        if self.checkpoint_dir is not None:
            dict_options = {}
            if not self.save_all_checkpoints:
                dict_options['keep_period'] = 1

            options = checkpoint.CheckpointManagerOptions(**dict_options)

            self.checkpointer = checkpoint.CheckpointManager(
                self.checkpoint_dir,
                {'network_params': checkpoint.PyTreeCheckpointer(), 'optimizer_params': checkpoint.PyTreeCheckpointer()},
                options=options)

        self.episode_num = 0
        self.num_steps = 0

    def checkpoint(self, network_params: dict, optimizer_params: optax.Params):
        # TODO: potentially add more saving stuff here
        self.checkpointer.save(self.num_steps, {'network_params': network_params, 'optimizer_params': optimizer_params})

    def episode_stat_string(self, episode_reward: float, avg_episode_loss: float, t: int,
                           additional_info: dict = None):
        print_str = (f"Episode {self.episode_num}, steps: {t + 1}, "
                    f"total steps: {self.num_steps}, "
                    f"rewards: {episode_reward:.2f}, "
                    f"avg episode loss: {avg_episode_loss:.8f}")

        if additional_info is not None:
            print_str += ", "
            for k, v in additional_info.items():
                print_str += f"{k}: {v / (t + 1):.4f}, "
        return print_str

    def add_returns_to_batches(self, episode_rewards: List[float], episode_batches: List[Batch]) -> List[Batch]:
        episode_rewards = jnp.array(episode_rewards)
        episode_discounts = jnp.array([(1 - b.done) * self.discounting for b in episode_batches])

        discounted_rewards = episode_rewards * episode_discounts
        overdiscounted_returns = jnp.cumsum(discounted_rewards[::-1])[::-1]
        returns = overdiscounted_returns / jnp.maximum(episode_discounts, 1e-10)

        for g, batch in zip(returns, episode_batches):
            batch.returns = g

        return episode_batches

    def sample_and_update(self, network_params: dict, optimizer_params: optax.Params) \
            -> Tuple[dict, optax.Params, dict]:
        seq_len = self.agent.trunc
        if self.online_training:  # online training
            sample = self.buffer.sample_idx(np.arange(len(self.buffer))[None, :])
        else:
            # sample a sequence from our buffer!
            sample = self.buffer.sample(self.batch_size, seq_len=seq_len)

        sample.gamma = (1 - sample.done) * self.discounting

        network_params, optimizer_params, info = self.agent.update(network_params, optimizer_params, sample)
        return network_params, optimizer_params, info

    def evaluate(self, network_params: dict) -> dict:
        test_info, self._rand_key = test_episodes(self.agent, network_params, self.env, self._rand_key,
                                                  n_episodes=self.offline_eval_episodes,
                                                  test_eps=self.offline_eval_epsilon,
                                                  action_cond=self.action_cond,
                                                  max_episode_steps=self.max_episode_steps)
        return test_info

    def train(self):
        all_logs = {'episode_infos': [], 'offline_eval': []}
        pbar = tqdm(total=self.total_steps, position=0, leave=True)

        network_params, optimizer_params, self._rand_key = self.agent.init_params(self._rand_key)

        while self.num_steps < self.total_steps:
            episode_reward, episode_batches = [], []
            episode_info = {'total_episode_loss': 0, 'episode_updates': 0}

            checkpoint_after_ep = False

            # get new initial hidden state
            prev_hs, self._rand_key = self.agent.reset(self._rand_key)

            # We do online training by resetting the buffer after every episode
            if self.online_training:
                self.buffer.reset()

            obs, env_info = self.env.reset()
            if self.one_hot_obses:
                obs = one_hot(obs, self.env.n_obs)

            if self.action_cond == 'cat':  # Action conditioning for t=-1 action
                action_encoding = np.zeros(self.env.n_actions)
                obs = np.concatenate([obs, action_encoding], axis=-1)

            action, self._rand_key, hs, qs = self.agent.act(network_params, obs, prev_hs, self._rand_key)
            action = action.item()

            for t in range(self.max_episode_steps):
                next_obs, reward, done, _, info = self.env.step(action, gamma_terminal=self.gamma_terminal)
                if self.one_hot_obses:
                    next_obs = one_hot(next_obs, self.env.n_obs)

                if self.action_cond == 'cat':  # Action conditioning
                    action_encoding = one_hot(action, self.env.n_actions)
                    next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                next_action, self._rand_key, next_hs, qs = self.agent.act(network_params, next_obs, hs, self._rand_key)
                next_action = next_action.item()

                batch = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                              next_action=next_action, state=prev_hs[0], next_state=hs[0],
                              end=done or (t == self.max_episode_steps - 1))

                episode_reward.append(reward)

                # Defer adding batch to buffer until episode's end, after we calculate returns.
                if self.include_returns_in_batch:
                    episode_batches.append(batch)
                else:
                    self.buffer.push(batch)

                # If we have enough things in the buffer to update
                if self.batch_size < len(self.buffer) and self.trunc < len(self.buffer):
                    network_params, optimizer_params, info = self.sample_and_update(network_params, optimizer_params)

                    episode_info['total_episode_loss'] += info['total_loss']
                    episode_info['episode_updates'] += 1

                pbar.update(1)
                self.num_steps += 1

                # Offline evaluation
                if self.offline_eval_freq is not None and self.offline_eval_freq % self.num_steps == 0:
                    test_info = self.evaluate(network_params)
                    all_logs['offline_eval'].append(test_info)

                if done:
                    break

                # bump time step
                prev_hs, hs = hs, next_hs
                obs, action = next_obs, next_action

                if self.checkpoint_dir is not None and self.checkpoint_freq > 0 and \
                        self.num_steps % self.checkpoint_freq == 0:
                    checkpoint_after_ep = True

            if self.include_returns_in_batch:
                batch_with_returns = self.add_returns_to_batches(episode_reward, episode_batches)
                for b in batch_with_returns:
                    self.buffer.push(b)

            # Online MC training
            if self.online_training and self.include_returns_in_batch:
                network_params, optimizer_params, info = self.sample_and_update(network_params, optimizer_params)
                episode_info['total_episode_loss'] += info['total_loss']
                episode_info['episode_updates'] += 1

            episode_info |= compress_episode_rewards(episode_reward)
            all_logs['episode_infos'].append(episode_info)

            self.episode_num += 1

            avg_episode_loss = episode_info['total_episode_loss'] / max(episode_info['episode_updates'], 1)
            print(self.episode_stat_string(sum(episode_reward), avg_episode_loss, t))

            if self.checkpoint_dir is not None and checkpoint_after_ep:
                self.checkpoint(network_params, optimizer_params)

        if self.checkpoint_dir is not None:
            self.checkpoint(network_params, optimizer_params)

        return network_params, optimizer_params, all_logs
