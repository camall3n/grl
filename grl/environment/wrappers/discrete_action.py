from typing import Union, List

import gymnasium as gym
import numpy as np

class ContinuousToDiscrete(gym.ActionWrapper):
    """
    Taken (and modified) from https://github.com/thu-ml/tianshou/blob/master/tianshou/env/gym_wrappers.py
    Gym environment wrapper to take discrete action in a continuous environment.

    :param gym.Env env: gym environment with continuous action space.
    :param int action_per_dim: number of discrete actions in each dimension
        of the action space.
    """

    def __init__(self, env: gym.Env, action_per_dim: Union[int, List[int]]) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        low, high = env.action_space.low, env.action_space.high
        if isinstance(action_per_dim, int):
            action_per_dim = [action_per_dim] * env.action_space.shape[0]
        assert len(action_per_dim) == env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(action_per_dim)
        if len(action_per_dim) == 1:
            self.action_space = gym.spaces.Discrete(action_per_dim[0])
        self.mesh = np.array(
            [np.linspace(lo, hi, a) for lo, hi, a in zip(low, high, action_per_dim)],
            dtype=object
        )

    def action(self, act: Union[np.ndarray, int]) -> np.ndarray:
        if isinstance(act, int):
            return np.array([self.mesh[0][act]])

        if len(act.shape) == 1:
            return np.array([self.mesh[i][a] for i, a in enumerate(act)])
        return np.array([[self.mesh[i][a] for i, a in enumerate(a_)] for a_ in act])
