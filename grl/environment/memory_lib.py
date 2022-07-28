import numpy as np

### 1 bit memory functions with three obs: r, b, t

memory_0 = np.array([
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

memory_3 = np.array([
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

memory_4 = np.array([
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

memory_12 = np.array([
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

memory_13 = np.array([
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

memory_14 = np.array([
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