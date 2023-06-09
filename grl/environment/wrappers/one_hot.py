from typing import Union, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from grl.utils.data import one_hot

class OneHotObservationWrapper(gym.Wrapper):
    """
    One-hot observation wrapper.
    Assumes that the env passed into this wrapper has discrete-integer valued observations.
    returns observations that are one-hot encodings of the discrete integer observation.
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]):
        super().__init__(env)
        assert isinstance(self.env.observation_space, spaces.Discrete), \
            "Cannot call One-hot wrapper on non-discrete observation space."

    @property
    def observation_space(self) -> spaces.MultiBinary:
        return spaces.MultiBinary(self.env.observation_space.n)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs_idx, info = self.env.reset()
        observation = one_hot(obs_idx, self.env.observation_space.n)
        return observation, info

    def step(self, action: int, **kwargs):
        obs_idx, reward, terminal, truncated, info = self.env.step(action, **kwargs)

        observation = one_hot(obs_idx, self.env.observation_space.n)
        return observation, reward, terminal, truncated, info

class OneHotActionConcatWrapper(gym.Wrapper):
    """
    One-hot action concatenation wrapper for observations.
    returns observations with a one-hot encoding of the current action concatenated.
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]):
        super().__init__(env)
        assert isinstance(self.env.action_space, spaces.Discrete)

    @property
    def observation_space(self) -> spaces.Space:
        obs_space = self.env.observation_space
        if isinstance(obs_space, spaces.Discrete):
            return spaces.MultiDiscrete(1 + self.action_space.n)
        elif isinstance(obs_space, spaces.MultiBinary):
            return spaces.MultiBinary(obs_space.n + self.action_space.n)
        else:
            return NotImplementedError

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # If we are at the start of an episode,
        # our action encoding is just a vector of zeros
        action_encoding = np.zeros(self.env.action_space.n)
        observation, info = self.env.reset()
        action_concat_obs = np.concatenate([observation, action_encoding])

        return action_concat_obs, info

    def step(self, action: int, **kwargs) -> Tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminal, truncated, info = self.env.step(action, **kwargs)

        action_encoding = one_hot(action, self.action_space.n)
        action_concat_obs = np.concatenate([observation, action_encoding])

        return action_concat_obs, reward, terminal, truncated, info
