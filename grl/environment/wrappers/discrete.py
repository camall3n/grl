import numpy as np
from typing import Union, Tuple
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete

class DiscreteObservationWrapper(gym.Wrapper):
    """
    A wrapper for a gym env for discrete observations.
    The wrapper wraps the discrete observation in a
    single element, 1D numpy array.
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]) -> None:
        super().__init__(env)
        assert isinstance(self.env.observation_space, Discrete)

        self.observation_space = MultiDiscrete([self.env.observation_space.n])

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        observation, info = self.env.reset(**kwargs)
        return np.array([observation]), info

    def step(self, action: int, **kwargs) -> Tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminal, truncated, info = self.env.step(action, **kwargs)

        return np.array([observation]), reward, terminal, truncated, info

