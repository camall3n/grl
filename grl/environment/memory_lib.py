import numpy as np
from itertools import product

from grl.utils.math import glorot_init, reverse_softmax
"""
1 bit memory functions with three obs: r, b, t
and 2 actions: up, down

Dimensions: AxZxMxM

"f" - fuzzy identity memory function
"""

def get_memory(memory_id: str,
               n_obs: int = None,
               n_actions: int = None,
               n_mem_states: int = 2,
               leakiness: float = 0.1) -> np.ndarray:
    current_module = globals()
    mem_name = f'memory_{memory_id}'
    if memory_id.isdigit():
        if int(memory_id) == 0:
            if n_obs is None or n_actions is None:
                raise ValueError('Must specify n_obs and n_actions when generating memory params')
            mem_params = glorot_init((n_actions, n_obs, n_mem_states, n_mem_states))
        else:
            if mem_name in current_module:
                T_mem = current_module[mem_name]
                # smooth out for softmax
                mem_params = reverse_softmax(T_mem)
            else:
                raise NotImplementedError(f'{mem_name} not found in memory_lib.py') from None
    elif memory_id == 'f':
        assert (n_obs is not None) and (n_actions is not None), \
            f"Either arguments n_obs and n_actions cannot be None for glorot_init. Got {n_obs} and {n_actions} respectively."
        identity = np.eye(n_mem_states)
        fuzzy_identity = identity - identity * leakiness + (1 - identity) * leakiness
        # 1 x 1 x n_mem_states x n_mem_states
        fuzzy_expanded_identity = np.expand_dims(np.expand_dims(fuzzy_identity, 0), 0)
        mem_func = fuzzy_expanded_identity.repeat(n_obs, axis=1).repeat(n_actions, axis=0)

        mem_params = reverse_softmax(mem_func)
    else:
        raise NotImplementedError(f"No memory of id {memory_id} exists.")
    return mem_params

def all_n_state_deterministic_memory(n_mem_states: int):
    id = np.eye(n_mem_states)
    idxes = [list(range(n_mem_states))]
    all_idxes = list(product(*(n_mem_states * idxes)))
    all_mem_funcs = id[all_idxes]
    return all_mem_funcs

mem_1 = np.array([
    [ # red
        # Pr(m'| m, o)
        # m0', m1'
        [1., 0], # m0
        [1, 0], # m1
    ],
    [ # blue
        [1, 0],
        [1, 0],
    ],
    [ # terminal
        [1, 0],
        [1, 0],
    ],
])
memory_1 = np.array([mem_1, mem_1]) # up, down

mem_3 = np.array([
    [ # red
        [1., 0], # s0, s1
        [0, 1],
    ],
    [ # blue
        [1, 0],
        [0, 1],
    ],
    [ # terminal
        [1, 0],
        [0, 1],
    ],
])
memory_3 = np.array([mem_3, mem_3])

mem_4 = np.array([
    [ # red
        [1., 0], # s0, s1
        [1, 0],
    ],
    [ # blue
        [0, 1],
        [1, 0],
    ],
    [ # terminal
        [1, 0],
        [1, 0],
    ],
])
memory_4 = np.stack([mem_4, mem_4])

mem_5 = np.array([
    [ # red
        #s0, s1
        [1, 0.],
        [1, 0],
    ],
    [ # blue
        [0, 1],
        [0, 1],
    ],
    [ # terminal
        [1, 0],
        [1, 0],
    ],
])
memory_5 = np.stack([mem_5, mem_5])

mem_6 = np.array([
    [ # red
        #s0, s1
        [1, 0.],
        [0, 1],
    ],
    [ # blue
        [0, 1],
        [1, 0],
    ],
    [ # terminal
        [1, 0],
        [1, 0],
    ],
])
memory_6 = np.stack([mem_6, mem_6])

mem_7 = np.array([
    # "Have I ever seen blue?"
    [ # red
        #s0, s1
        [1, 0.],
        [0, 1],
    ],
    [ # blue
        [0, 1],
        [0, 1],
    ],
    [ # terminal
        [1, 0],
        [0, 1],
    ],
])
memory_7 = np.stack([mem_7, mem_7])

mem_12 = np.array([
    # always flip the bit!
    [ # red
        #s0, s1
        [0, 1.],
        [1, 0],
    ],
    [ # blue
        [0, 1],
        [1, 0],
    ],
    [ # terminal
        [0, 1],
        [1, 0],
    ],
])
memory_12 = np.stack([mem_12, mem_12])

mem_102 = np.array([
    # always flip the bit!
    [ # red
        #s0, s1
        [0, 1.],
        [1, 0],
    ],
    [ # terminal
        [0, 1],
        [1, 0],
    ],
])
memory_102 = np.stack([mem_102, mem_102])

mem_103 = np.array([
    # -> 1, always
    [ # red
        #m0' m1'
        [0, 1.],
        [0, 1],
    ],
    [ # terminal
        [0, 1],
        [0, 1],
    ],
])
memory_103 = np.stack([mem_103, mem_103])

mem_13 = np.array([
    [ # red
        [0., 1], # s0, s1
        [1, 0],
    ],
    [ # blue
        [0, 1],
        [0, 1],
    ],
    [ # terminal
        [0, 1],
        [1, 0],
    ],
])
memory_13 = np.stack([mem_13, mem_13])

mem_14 = np.array([
    [ # red
        [0., 1], # s0, s1
        [0, 1],
    ],
    [ # blue
        [0, 1],
        [1, 0],
    ],
    [ # terminal
        [0, 1],
        [1, 0],
    ],
])
memory_14 = np.stack([mem_14, mem_14])

mem_15_solid = np.array([
    [ # red
        [0., 1], # s0, s1
        [1, 0],
    ],
    [ # blue
        [1, 0],
        [1, 0],
    ],
    [ # terminal
        [1, 0],
        [1, 0],
    ],
])
mem_15_dashed = np.array([
    [ # red
        [1., 0], # s0, s1
        [1, 0],
    ],
    [ # blue
        [1, 0],
        [1, 0],
    ],
    [ # terminal
        [1, 0],
        [1, 0],
    ],
])
memory_15 = np.stack([mem_15_solid, mem_15_dashed])

# Optimal memory for t-maze
mem_16 = np.array([
    [
        # Pr(m'| m, o)
        # m0', m1'
        [1., 0], # m0
        [1, 0], # m1
    ],
    [ # we see the goal as DOWN
        [0, 1],
        [0, 1],
    ],
    [ # corridor
        [1, 0],
        [0, 1],
    ],
    [ # junction
        [1, 0],
        [0, 1],
    ],
    [ # terminal
        [1, 0],
        [0, 1],
    ],
])
memory_16 = np.array([mem_16, mem_16, mem_16, mem_16]) # up, down, right, left

# Memory for t-maze, where we can set the initial transition probabilities.
p = 0.4
q = 0.6
mem_17 = np.array([
    [ # we see the goal as UP
        # Pr(m'| m, o)
        # m0', m1'
        [p, 1 - p], # m0
        [q, 1 - q], # m1
    ],
    [ # we see the goal as DOWN
        [p, 1 - p],
        [q, 1 - q],
    ],
    [ # corridor
        [p, 1 - p],
        [q, 1 - q],
    ],
    [ # junction
        [p, 1 - p],
        [q, 1 - q],
    ],
    [ # terminal
        [p, 1 - p],
        [q, 1 - q],
    ],
])
memory_17 = np.array([mem_17, mem_17, mem_17, mem_17]) # up, down, right, left

# Optimal memory for t-maze
mem_18 = np.array([
    [ # we see the goal as UP
        # Pr(m'| m, o)
        # m0', m1'
        [1., 0], # m0
        [1, 0], # m1
    ],
    [ # we see the goal as DOWN
        [0, 1],
        [0, 1],
    ],
    [ # corridor
        [0, 1],
        [1, 0],
    ],
    [ # junction
        [1, 0],
        [0, 1],
    ],
    [ # terminal
        [1, 0],
        [0, 1],
    ],
])
memory_18 = np.array([mem_18, mem_18, mem_18, mem_18]) # up, down, right, left

memory_19 = np.zeros_like(memory_18) + 1 / memory_18.shape[-1]
