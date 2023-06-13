from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Union, List, Tuple
import warnings

from jax import random
import numpy as np
import optax
from orbax import checkpoint
from tqdm import tqdm

from grl.evaluation import eval_episodes
from grl.mdp import MDP, POMDP
from grl.agent.rnn import RNNAgent
from grl.utils.data import Batch, one_hot, compress_episode_rewards
from grl.utils.mdp import all_t_discounted_returns
from grl.utils.replaymemory import EpisodeBuffer

class Trainer:
    def __init__(self,
                 env: Union[MDP, POMDP],
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
        self.total_steps = self.args.total_steps

        # For RNNs
        self.trunc = self.args.trunc
        self.action_cond = self.args.action_cond

        # we train at the end of an episode only if we also need to learn an MC head.
        self.include_returns_in_batch = self.args.algo == 'multihead_rnn' and \
                                        self.args.multihead_loss_mode in ['both', 'mc', 'split']

        # Reward normalization
        self.reward_scale = 1.
        self.normalize_rewards = self.args.normalize_rewards
        if self.normalize_rewards:
            assert hasattr(self.env, 'R_max') and hasattr(self.env, 'R_min')
            self.reward_scale = 1 / (self.env.R_max - self.env.R_min)

        # Logging and eval
        self.offline_eval_freq = self.args.offline_eval_freq
        self.offline_eval_episodes = self.args.offline_eval_episodes
        self.offline_eval_epsilon = self.args.offline_eval_epsilon
        self.checkpoint_freq = self.args.checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_all_checkpoints = self.args.save_all_checkpoints

        self.batch_size = self.args.batch_size
        self.online_training = self.args.replay_size < 1

        # Replay buffer initialization
        replay_capacity = self.args.replay_size
        if replay_capacity < self.max_episode_steps:
            warnings.warn("Replay capacity is < max_episode_steps. "
                          f"Increasing to max_episode_steps={self.max_episode_steps}")
            replay_capacity = self.max_episode_steps

        # For the special case of LSTMs, we have both hidden AND cell states
        state_size = (self.args.hidden_size, )
        if self.args.arch == 'lstm':
            state_size = (2, ) + state_size

        self.buffer = EpisodeBuffer(replay_capacity,
                                    rand_key,
                                    self.env.observation_space.shape,
                                    state_size=state_size,
                                    unpack_state=self.args.arch == 'lstm')

        # Initialize checkpointers
        if self.checkpoint_dir is not None:
            dict_options = {}
            if not self.save_all_checkpoints:
                dict_options['keep_period'] = 1

            options = checkpoint.CheckpointManagerOptions(**dict_options)

            self.checkpointer = checkpoint.CheckpointManager(self.checkpoint_dir, {
                'network_params': checkpoint.PyTreeCheckpointer(),
                'optimizer_params': checkpoint.PyTreeCheckpointer()
            },
                                                             options=options)

        self.episode_num = 0
        self.num_steps = 0

    def checkpoint(self, network_params: dict, optimizer_params: optax.Params):
        # TODO: potentially add more saving stuff here
        self.checkpointer.save(self.num_steps, {
            'network_params': network_params,
            'optimizer_params': optimizer_params
        })

    def episode_stat_string(self,
                            episode_reward: float,
                            avg_episode_loss: float,
                            t: int,
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

    def add_returns_to_batches(self, episode_rewards: List[float],
                               episode_experiences: List[Batch]) -> List[Batch]:
        episode_rewards = np.array(episode_rewards)
        if self.normalize_rewards:
            episode_rewards *= self.reward_scale

        # Calculate discounts for every step
        discounts = np.array([(1 - b.done) * self.discounting for b in episode_experiences])

        returns = all_t_discounted_returns(discounts, episode_rewards)

        return [replace(batch, returns=g) for g, batch in zip(returns, episode_experiences)]

    def sample_and_update(self, network_params: dict, optimizer_params: optax.Params) \
            -> Tuple[dict, optax.Params, dict]:
        seq_len = self.agent.trunc
        if self.online_training: # online training
            sample = self.buffer.sample_idx(np.arange(len(self.buffer))[None, :])
        else:
            # sample a sequence from our buffer!
            sample = self.buffer.sample(self.batch_size, seq_len=seq_len)

        reward = sample.reward
        if self.normalize_rewards:
            reward = sample.reward * self.reward_scale

        gamma = (1 - sample.done) * self.discounting
        sample = replace(sample, gamma=gamma, reward=reward)

        network_params, optimizer_params, info = self.agent.update(network_params,
                                                                   optimizer_params, sample)
        return network_params, optimizer_params, info

    def evaluate(self, network_params: dict) -> dict:
        prev_state = self.env.current_state
        test_info, self._rand_key = eval_episodes(self.agent,
                                                  network_params,
                                                  self.env,
                                                  self._rand_key,
                                                  n_episodes=self.offline_eval_episodes,
                                                  test_eps=self.offline_eval_epsilon,
                                                  max_episode_steps=self.max_episode_steps)
        self.env.current_state = prev_state
        return test_info

    def train(self):
        all_logs = {'online_info': {}, 'offline_eval': []}
        pbar = tqdm(total=self.total_steps, position=0, leave=True)

        network_params, optimizer_params, self._rand_key = self.agent.init_params(self._rand_key)

        while self.num_steps < self.total_steps:
            episode_reward, episode_experiences = [], []
            episode_info = {'total_episode_loss': 0, 'episode_updates': 0}

            checkpoint_after_ep = False

            # get new initial hidden state
            prev_hs, self._rand_key = self.agent.reset(self._rand_key)

            # We do online training by resetting the buffer after every episode
            if self.online_training:
                self.buffer.reset()

            obs, env_info = self.env.reset()

            action, self._rand_key, hs, qs = self.agent.act(network_params, obs, prev_hs,
                                                            self._rand_key)
            action = action.item()

            for t in range(self.max_episode_steps):
                next_obs, reward, done, _, info = self.env.step(action,
                                                                gamma_terminal=self.gamma_terminal)
                next_action, self._rand_key, next_hs, qs = self.agent.act(
                    network_params, next_obs, hs, self._rand_key)
                next_action = next_action.item()

                experience = Batch(obs=obs,
                                   reward=reward,
                                   next_obs=next_obs,
                                   action=action,
                                   done=done,
                                   next_action=next_action,
                                   state=prev_hs[0],
                                   next_state=hs[0],
                                   end=done or (t == self.max_episode_steps - 1))

                episode_reward.append(reward)

                # Defer adding experience to buffer until episode's end, after we calculate returns.
                if self.include_returns_in_batch:
                    episode_experiences.append(experience)
                else:
                    self.buffer.push(experience)

                # If we have enough things in the buffer to update
                if self.batch_size < len(self.buffer) and self.trunc < len(self.buffer):
                    network_params, optimizer_params, info = self.sample_and_update(
                        network_params, optimizer_params)

                    episode_info['total_episode_loss'] += info['total_loss']
                    episode_info['episode_updates'] += 1

                pbar.update(1)
                self.num_steps += 1

                # Offline evaluation
                if self.offline_eval_freq is not None and self.num_steps % self.offline_eval_freq == 0:
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
                batch_with_returns = self.add_returns_to_batches(episode_reward,
                                                                 episode_experiences)
                for b in batch_with_returns:
                    self.buffer.push(b)

            # Online MC training
            if self.online_training and self.include_returns_in_batch:
                network_params, optimizer_params, info = self.sample_and_update(
                    network_params, optimizer_params)
                episode_info['total_episode_loss'] += info['total_loss'].item()
                episode_info['episode_updates'] += 1

            episode_info |= compress_episode_rewards(episode_reward)

            # save all our episode info in lists
            for k, v in episode_info.items():
                if k not in all_logs['online_info']:
                    all_logs['online_info'][k] = []
                all_logs['online_info'][k].append(v)

            self.episode_num += 1

            avg_episode_loss = episode_info['total_episode_loss'] / max(
                episode_info['episode_updates'], 1)
            print(self.episode_stat_string(sum(episode_reward), avg_episode_loss, t))

            if self.checkpoint_dir is not None and checkpoint_after_ep:
                self.checkpoint(network_params, optimizer_params)

        # at the end of training, save a checkpoint
        if self.checkpoint_dir is not None:
            self.checkpoint(network_params, optimizer_params)

        if 'total_loss' in all_logs:
            all_logs['total_loss'] = np.array(all_logs['total_loss'], dtype=np.half)

        return network_params, optimizer_params, all_logs
