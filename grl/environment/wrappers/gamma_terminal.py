import numpy as np
from typing import Union, Tuple
import gymnasium as gym

class GammaTerminalWrapper(gym.Wrapper):
    """
    A wrapper for a gymnasium.Env that makes it compatible with our step() API,
    in particular the gamma_terminal feature. 
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper], gamma: float = None) -> None:
        super().__init__(env)
        if gamma is None:
            if hasattr(self.env, 'gamma'):
                self.gamma_terminal_prob = 1. - self.env.gamma
                self.gamma = 1.
            else:
                raise AttributeError(
                    "If environment does not come with a gamma, need to set gamma for gamma termination"
                )
        else:
            self.gamma_terminal_prob = 1. - gamma
            self.gamma = 1.

    def step(self, action: int, **kwargs) -> Tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminal, truncated, info = self.env.step(action, **kwargs)
        # if the env happens to have a rand_key, use that for reproducibility.
        # Otherwise, just use random.uniform.
        # (By the way, this will not error because python evaluates boolean statements lazily.)
        if hasattr(self.env, 'rand_key') and self.env.rand_key is not None:
            unif = self.rand_key.uniform()
        else:
            unif = np.random.uniform()
        if unif < self.gamma_terminal_prob:
            terminal = True

        return observation, reward, terminal, truncated, info
