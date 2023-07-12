import numpy as np
from typing import Union, Tuple
import gymnasium as gym

class TupleObservationWrapper(gym.Wrapper):
    """
    A wrapper for a gym env for tuple-of-discrete observations.
    The wrapper wraps the tuple in a 1D numpy array.
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]) -> None:
        super().__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Tuple)
        obs_space = []

        for sp in self.env.observation_space:
            assert isinstance(sp, gym.spaces.Discrete)
            obs_space.append(sp.n)

        self.observation_space = gym.spaces.MultiDiscrete(obs_space)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        observation, info = self.env.reset(**kwargs)
        return np.array(observation), info

    def step(self, action: int, **kwargs) -> Tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminal, truncated, info = self.env.step(action, **kwargs)

        return np.array(observation), reward, terminal, truncated, info

