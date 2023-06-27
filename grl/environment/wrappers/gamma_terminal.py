import numpy as np
from typing import Union, Tuple
from grl.mdp import MDP
import gymnasium as gym

class GammaTerminalWrapper(gym.Wrapper):
    """
    A wrapper for a gymnasium.Env that makes it compatible with our step() API,
    in particular the gamma_terminal feature. 
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper], gamma) -> None:
        super().__init__(env)
        assert not isinstance(self.env, MDP)
        self.gamma = gamma

    def step(self, action: int, gamma_terminal: bool, **kwargs) -> Tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminal, truncated, info = self.env.step(action, **kwargs)
        if gamma_terminal:
            # if the env happens to have a rand_key, use that for reproducibility. 
            # Otherwise, just use random.uniform.
            # (By the way, this will not error because python evaluates boolean statements lazily.)
            if hasattr(self.env, 'rand_key') and self.env.rand_key is not None:
                unif = self.rand_key.uniform()
            else:
                unif = np.random.uniform()
            if unif < (1 - self.gamma):
                terminal = True
            
        return observation, reward, terminal, truncated, info
