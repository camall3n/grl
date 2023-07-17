import numpy as np
from typing import Union, Tuple
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete

class FlattenMultiDiscreteActionWrapper(gym.ActionWrapper):
    """
    A wrapper for a gym env for multi-discrete actions.
    It flattens this action space into a gym.spaces.Discrete action space
    """
    def __init__(self, env: Union[gym.Env, gym.Wrapper]) -> None:
        super().__init__(env)
        assert isinstance(self.env.action_space, MultiDiscrete)
        n_total_actions = 1
        for act_space in self.env.action_space:
            n_total_actions *= act_space.n

        self.action_space = Discrete(int(n_total_actions))

    def action(self, action: int) -> np.ndarray:
        multi_action = []
        act_so_far = action
        for act_space in self.env.action_space[:-1]:
            multi_action.append(act_so_far // act_space.n)
            act_so_far = act_so_far % act_space.n

        multi_action.append(act_so_far)
        return np.array(multi_action)
