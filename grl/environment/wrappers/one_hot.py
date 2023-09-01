from typing import Union, Tuple

import gymnasium as gym
from gymnasium import spaces
from popgym.wrappers import PreviousAction
import numpy as np

from grl.utils.data import one_hot

class OneHotObservationWrapper(gym.ObservationWrapper):
    """
    One-hot observation wrapper.
    Assumes that the env passed into this wrapper has discrete-integer valued observations.
    returns observations that are one-hot encodings of the discrete integer observation.
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]):
        super().__init__(env)
        assert isinstance(self.env.observation_space, spaces.Discrete), \
            "Cannot call One-hot wrapper on non-discrete observation space."

    def observation(self, observation: int) -> np.ndarray:
        return one_hot(observation, self.env.observation_space.n)

    @property
    def observation_space(self) -> spaces.MultiBinary:
        return spaces.MultiBinary(self.env.observation_space.n)

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
        elif isinstance(obs_space, spaces.Box):
            return spaces.Box(low=np.concatenate(
                [obs_space.low,
                 np.zeros(self.action_space.n, dtype=obs_space.low.dtype)]),
                              high=np.concatenate([
                                  obs_space.high,
                                  np.ones(self.action_space.n, dtype=obs_space.low.dtype)
                              ]))
        else:
            return NotImplementedError

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # If we are at the start of an episode,
        # get null encoding based on the environment
        action_encoding = one_hot(PreviousAction.get_null_action(self.env.action_space),
                                  self.action_space.n)
        observation, info = self.env.reset(**kwargs)
        action_concat_obs = np.concatenate([observation, action_encoding])
        return action_concat_obs, info

    def step(self, action: int, **kwargs) -> Tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminal, truncated, info = self.env.step(action, **kwargs)

        action_encoding = one_hot(action, self.action_space.n)
        action_concat_obs = np.concatenate([observation, action_encoding])

        return action_concat_obs, reward, terminal, truncated, info
