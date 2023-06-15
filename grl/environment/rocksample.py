import json
import gymnasium as gym
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, random, vmap
from typing import Tuple
from itertools import product
from pathlib import Path

from grl.utils.math import euclidian_dist

from definitions import ROOT_DIR

def half_dist_prob(dist: float, max_dist: float, lb: float = 0.5):
    prob = (1 + jnp.power(2.0, -dist / max_dist)) * lb
    return prob

class RockSample(gym.Env):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self,
                 config_file: Path = None,
                 rand_key: jax.random.PRNGKey = None,
                 rock_obs_init: float = 0.,
                 **kwargs):
        """
        RockSample environment.
        Observations: position (2) and rock goodness/badness
        Actions: k + 5 actions. Actions are as follows:
        0: North, 1: East, 2: South, 3: West,
        4: Sample,
        5: Check rock 1, ..., k + 5: Check rock k
        :param config_file: config file for RockSample (located in unc/envs/configs)
        :param seed: random seed for the environment.
        :param rock_obs_init: What do we initialize our rock observations with?
        """
        if config_file is None:
            config_file = Path(ROOT_DIR, 'grl', 'environment', 'configs',
                               'rocksample_7_8_config.json')

        if rand_key is None:
            rand_key = random.PRNGKey(np.random.randint(1, 100000))

        with open(config_file) as f:
            config = json.load(f)

        self.size = config['size']
        self.k = config['rocks']
        self.gamma = config['gamma']
        self.half_efficiency_distance = config['half_efficiency_distance']
        self.bad_rock_reward = config['bad_rock_reward']
        self.good_rock_reward = config['good_rock_reward']
        self.exit_reward = config['exit_reward']

        all_rewards = [self.bad_rock_reward, self.good_rock_reward, self.exit_reward]
        self.R_max = max(all_rewards)
        self.R_min = min(all_rewards)

        self.observation_space = gym.spaces.MultiBinary(self.size**2 + self.k)
        self.state_space = gym.spaces.MultiDiscrete(2 + 3 * self.k)
        self.action_space = gym.spaces.Discrete(self.k + 5)

        # This helps with saving
        self.unique_rewards = {0}
        self.unique_rewards.add(self.good_rock_reward)
        self.unique_rewards.add(self.bad_rock_reward)
        self.unique_rewards.add(self.exit_reward)
        self.unique_rewards = list(self.unique_rewards)

        self.rand_key = rand_key
        self.rock_obs_init = rock_obs_init
        self.position_max = np.array([self.size - 1, self.size - 1])
        self.position_min = np.array([0, 0])

        self._move_transition = jit(self._move_transition, static_argnames=['action'])
        self._check_transition = jit(self._check_transition,
                                     static_argnames=['half_efficiency_distance', 'action'])
        self._sample_transition = jit(self._sample_transition)
        # self.generate_map = jit(self.generate_map, static_argnames=['size', 'k'])
        self.sample_morality = jit(self.sample_morality, static_argnames=['k'])
        self.sample_positions = jit(self.sample_positions, static_argnames=['size', 'k'])

        self.rock_positions = None
        self.rock_morality = None
        self.agent_position = None
        self.sampled_rocks = None
        self.checked_rocks = None
        self.current_rocks_obs = jnp.zeros(self.k) + rock_obs_init

        # Given a random seed, generate the map
        """
        TODO: refactor this so that rock positions are passed in, and sampled
        outside of the environment init.
        This is due to the train and test envs generating different
        rock positions.
        """
        self.rock_positions, self.rand_key = self.generate_map(self.size, self.k, self.rand_key)

    @property
    def current_state(self):
        """
        return the underlying state of the environment.
        State consists of 3k + 2 features
        2 are the position
        k are for the underlying rock morality
        k are for which rocks were already sampled
        k are for what the current rock observation is
        IN THIS ORDER
        :return: underlying environment state
        """
        ap = self.agent_position.copy()
        rm = self.rock_morality.copy()
        sr = self.sampled_rocks.copy().astype(int)
        cro = self.current_rocks_obs.copy()
        return jnp.concatenate([ap, rm, sr, cro])

    @current_state.setter
    def current_state(self, state: np.ndarray):
        ap, rm, sr, cro = self.unpack_state(state)
        self.agent_position = ap
        self.rock_morality = rm
        self.sampled_rocks = sr
        self.current_rocks_obs = cro

    @staticmethod
    def generate_map(size: int, k: int,
                     rand_key: random.PRNGKey) -> Tuple[np.ndarray, random.PRNGKey]:
        positions_key, rand_key = random.split(rand_key)
        rows_range = np.arange(0, size)
        cols_range = rows_range[:-1]
        possible_rock_positions = np.array(list(product(rows_range, cols_range)), dtype=np.int16)
        all_positions_idx = random.choice(positions_key,
                                          possible_rock_positions.shape[0], (k, ),
                                          replace=False)
        all_positions = possible_rock_positions[all_positions_idx]
        return all_positions, rand_key

    @staticmethod
    def sample_positions(size: int, rand_key: random.PRNGKey, n: int = 1):
        position_key, rand_key = random.split(rand_key)
        rows_range = jnp.arange(0, size + 1)
        cols_range = rows_range[:-1]
        possible_positions = jnp.array(list(product(rows_range, cols_range)), dtype=np.int16)
        sample_idx = random.choice(position_key, possible_positions.shape[0], (n, ), replace=True)
        sample_position = jnp.array(possible_positions[sample_idx], dtype=np.int16)
        return sample_position, rand_key

    def sample_all_states(self, rock_morality: np.ndarray = None) -> np.ndarray:
        """
        Return observations from all possible non-terminal agent positions,
        with a given rock morality.
        :param rock_morality:
        :return:
        """
        if rock_morality is None:
            rock_morality = self.rock_morality

        states = []

        for y in range(self.size):
            for x in range(self.size - 1):
                pos = np.array([y, x])
                state = np.concatenate(
                    [pos, rock_morality, self.sampled_rocks, self.current_rocks_obs])
                states.append(state)

        return np.stack(states)

    @staticmethod
    def sample_morality(k: int, rand_key: random.PRNGKey) -> Tuple[jnp.ndarray, random.PRNGKey]:
        bit_key, rand_key = random.split(rand_key)
        return random.bernoulli(bit_key, shape=(k, )).astype(int), rand_key

    def get_terminal(self, gamma_terminal: bool = False) -> bool:
        terminal = self.agent_position[1] == (self.size - 1)
        if gamma_terminal:
            terminal |= self.get_gamma_terminal(self.rand_key, self.gamma)
        return terminal

    @staticmethod
    @jit
    def get_gamma_terminal(rand_key: random.PRNGKey, gamma: float):
        gamma_terminal_key, rand_key = random.split(rand_key)
        return not random.bernoulli(gamma_terminal_key, p=gamma)

    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def observe(self, state: np.ndarray) -> jnp.ndarray:
        """
        Observation is dependent on action here.
        Observation is a size ** 2 + k vector:
        first two features are the position.
        last k features are the current_rock_obs.
        :param state:
        :param action:
        :return:
        """
        position, _, _, rocks_obs = self.unpack_state(state)
        position_obs = np.zeros((self.size, self.size))
        position_obs[int(position[0]), int(position[1])] = 1
        position_obs = position_obs.flatten()

        return jnp.concatenate([position_obs, rocks_obs])

    def get_reward(self, prev_state: jnp.ndarray, action: int) -> int:
        """
        NOTE: this should be called AFTER transition happens
        :param action: action that you're planning on taking
        :return: reward
        """
        rew = 0

        if action == 4:
            # If we're SAMPLING
            ele = (self.rock_positions == self.agent_position)
            idx = np.nonzero(ele[:, 0] & ele[:, 1])[0]
            if idx.shape[0] > 0:
                idx = idx[0]
                # If we're on a rock, we get rewards by sampling accordingly
                _, prev_rock_morality, _, _ = self.unpack_state(prev_state)
                rew = self.good_rock_reward if prev_rock_morality[idx] > 0 else self.bad_rock_reward

        elif action < 4:
            # If we're MOVING
            if self.get_terminal():
                rew = self.exit_reward

        return rew

    def reset(self) -> Tuple[jnp.ndarray, dict]:
        self.sampled_rocks = jnp.zeros(len(self.rock_positions)).astype(bool)
        self.checked_rocks = jnp.zeros_like(self.sampled_rocks).astype(bool)
        self.rock_morality, self.rand_key = self.sample_morality(self.k, self.rand_key)
        agent_position, self.rand_key = self.sample_positions(1, self.rand_key)
        self.agent_position = agent_position[0]
        self.current_rocks_obs = np.zeros_like(self.current_rocks_obs) + self.rock_obs_init

        info = {
            'rock_morality': self.rock_morality,
            'agent_position': self.agent_position,
            'rock_positions': self.rock_positions
        }

        return self.observe(self.current_state), info

    def unpack_state(self,
                     state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = state[:2].astype(int)
        rock_morality = state[2:2 + self.k]
        sampled_rocks = state[2 + self.k:2 + 2 * self.k]
        current_rocks_obs = state[2 + 2 * self.k:]
        return position, rock_morality, sampled_rocks, current_rocks_obs

    @staticmethod
    def pack_state(position: np.ndarray, rock_morality: np.ndarray, sampled_rocks: np.ndarray,
                   current_rocks_obs: np.ndarray):
        return np.concatenate([position, rock_morality, sampled_rocks, current_rocks_obs])

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        action = actions[0]
        bs = states.shape[0]
        positions = states[:, :2]
        rock_moralities = states[:, 2:2 + self.k]
        sampled_rockses = states[:, 2 + self.k:2 + 2 * self.k]
        current_rocks_obses = states[:, 2 + 2 * self.k:]

        rock_positionses = np.repeat(np.expand_dims(self.rock_positions, 0), bs, axis=0)
        direction_mappings = np.repeat(np.expand_dims(self.direction_mapping, 0), bs, axis=0)
        position_maxes = np.repeat(np.expand_dims(self.position_max, 0), bs, axis=0)
        position_mins = np.repeat(np.expand_dims(self.position_min, 0), bs, axis=0)
        rand_keys = random.split(self.rand_key, num=bs + 1)
        self.rand_key, rand_keys = rand_keys[0], rand_keys[1:]

        if action > 4:
            # CHECK
            new_rocks_obses, rand_keys = vmap(self._check_transition)(current_rocks_obses,
                                                                      positions, rock_positionses,
                                                                      rock_moralities, rand_keys,
                                                                      actions)
            current_rocks_obses = new_rocks_obses
        elif action == 4:
            # SAMPLING
            new_sampled_rockses, new_rocks_obses, new_rock_moralities = vmap(
                self._sample_transition)(positions, sampled_rockses, current_rocks_obses,
                                         rock_moralities, rock_positionses)
            sampled_rockses, current_rocks_obses, rock_moralities = new_sampled_rockses, new_rocks_obses, new_rock_moralities

            # If we sample a space with no rocks, nothing happens for transition.
        else:
            # MOVING
            positions = vmap(self._move_transition)(positions, direction_mappings, position_maxes,
                                                    position_mins, actions)
        return np.concatenate([positions, rock_moralities, sampled_rockses, current_rocks_obses],
                              axis=1)

    @staticmethod
    def _check_transition(current_rocks_obs: np.ndarray, position: np.ndarray,
                          rock_positions: np.ndarray, rock_morality: np.ndarray,
                          rand_key: random.PRNGKey, half_efficiency_distance: float, action: int):
        rock_idx = action - 5
        dist = euclidian_dist(position, rock_positions[rock_idx])
        prob = half_dist_prob(dist, half_efficiency_distance)

        # w.p. prob we return correct rock observation.
        rock_obs = rock_morality[rock_idx]
        choices = jnp.array([rock_obs, 1 - rock_obs])
        probs = jnp.array([prob, 1 - prob])
        key, subkey = random.split(rand_key)
        rock_obs = random.choice(subkey, choices, (1, ), p=probs)[0]

        new_rocks_obs = current_rocks_obs.at[rock_idx].set(rock_obs)
        return new_rocks_obs, key

    @staticmethod
    def _move_transition(position: np.ndarray, direction_mapping: np.ndarray,
                         position_max: np.ndarray, position_min: np.ndarray, action: int):
        new_pos = position + direction_mapping[action]
        position = jnp.maximum(jnp.minimum(new_pos, position_max), position_min)
        return position

    @staticmethod
    def _sample_transition(position: jnp.ndarray, sampled_rocks: jnp.ndarray,
                           current_rocks_obs: jnp.ndarray, rock_mortality: jnp.ndarray,
                           rock_positions: jnp.ndarray):
        ele = (rock_positions == position)
        bool_pos = (ele[:, 0] & ele[:, 1]).astype(int)

        zero_arr = jnp.zeros_like(current_rocks_obs)
        new_sampled_rocks = jnp.minimum(sampled_rocks + bool_pos, zero_arr + 1)
        new_rocks_obs = jnp.maximum(current_rocks_obs - bool_pos, zero_arr)
        new_rock_morality = jnp.maximum(rock_mortality - bool_pos, zero_arr)
        return new_sampled_rocks, new_rocks_obs, new_rock_morality

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        position, rock_morality, sampled_rocks, current_rocks_obs = self.unpack_state(state)

        if action > 4:
            # CHECK
            new_rocks_obs, self.rand_key = self._check_transition(current_rocks_obs.copy(),
                                                                  position, self.rock_positions,
                                                                  rock_morality, self.rand_key,
                                                                  self.half_efficiency_distance,
                                                                  action)

            current_rocks_obs = new_rocks_obs
        elif action == 4:
            # SAMPLING
            new_sampled_rocks, new_rocks_obs, new_rock_morality = self._sample_transition(
                position, sampled_rocks, current_rocks_obs, rock_morality, self.rock_positions)
            sampled_rocks, current_rocks_obs, rock_morality = new_sampled_rocks, new_rocks_obs, new_rock_morality

            # If we sample a space with no rocks, nothing happens for transition.
        else:
            # MOVING
            position = self._move_transition(position, self.direction_mapping, self.position_max,
                                             self.position_min, action)

        return self.pack_state(position, rock_morality, sampled_rocks, current_rocks_obs)

    def emit_prob(self, states: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        With rock sample, we use an UNWEIGHTED particle filter
        (like in the POMCP paper). This function will simply return an array
        of shape (len(states),) of 1's
        :param states: size (batch_size, *state_shape)
        :param obs: size (batch_size, *obs_shape)
        :return: array of 1's of shape (batch_size, )
        """
        if len(states.shape) > 1:
            ones = jnp.ones(states.shape[0])
        else:
            ones = jnp.ones(1)
        return ones

    def step(self, action: int, gamma_terminal: bool = False):
        prev_state = self.current_state
        self.current_state = self.transition(self.current_state, action)

        rock_idx = action - 5
        if rock_idx >= 0:
            self.checked_rocks = self.checked_rocks.at[rock_idx].set(True)

        info = {
            'rock_morality': self.rock_morality,
            'agent_position': self.agent_position,
            'rock_positions': self.rock_positions
        }
        obs = self.observe(self.current_state)
        reward = self.get_reward(prev_state, action)
        terminal = self.get_terminal(gamma_terminal=gamma_terminal)
        truncated = False

        return obs, reward, terminal, truncated, info

    def generate_array(self) -> jnp.ndarray:
        """
        Generate numpy array representing state.
        Mappings are as follows:
        0 = white space
        1 = agent
        2 = rock
        3 = agent + rock
        4 = goal
        :return:
        """
        viz_array = np.zeros((self.size, self.size))

        viz_array[self.rock_positions[:, 0], self.rock_positions[:, 1]] = 2

        viz_array[int(self.agent_position[0]), int(self.agent_position[1])] += 1

        viz_array[:, self.size - 1] = 4
        return jnp.array(viz_array)
