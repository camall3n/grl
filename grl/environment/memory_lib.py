import numpy as np
"""
1 bit memory functions with three obs: r, b, t
and 2 actions: up, down

Dimensions: AxZxMxM
"""

mem_0 = np.array([
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
memory_0 = np.array([mem_0, mem_0]) # up, down

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