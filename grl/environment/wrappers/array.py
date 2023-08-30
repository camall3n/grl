import numpy as np
from typing import Union, Tuple
import gymnasium as gym
from gymnasium.spaces import Tuple, MultiDiscrete

class ArrayObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper for a gym env for discrete observations.
    The wrapper casts the multi discrete observation in a
    single element, 1D numpy array.
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]) -> None:
        super().__init__(env)
        assert isinstance(self.env.observation_space, Tuple)

        obs_space = []

        for sp in self.env.observation_space:
            assert isinstance(sp, gym.spaces.Discrete)
            obs_space.append(sp.n)

        self.observation_space = gym.spaces.MultiDiscrete(obs_space)

    def observation(self, observation: int) -> np.ndarray:
        return np.array(observation)
