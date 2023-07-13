import jax.numpy as jnp

from grl.utils.math import reverse_softmax

def tiger_alt_start_pi(**kwargs) -> jnp.ndarray:
    # actions are listen, open-left, open-right
    pi_phi = jnp.array([
        [1, 0, 0], # init
        [4 / 6, 1 / 6, 1 / 6], # tiger-left
        [2 / 5, 1 / 5, 2 / 5], # tiger-right
        [1, 0, 0], # terminal
    ])

    pi_params = reverse_softmax(pi_phi)
    return pi_params

def tiger_alt_start_uniform(**kwargs) -> jnp.ndarray:
    # actions are listen, open-left, open-right
    pi_phi = jnp.array([
        [1 / 3, 1 / 3, 1 / 3], # init
        [1 / 3, 1 / 3, 1 / 3], # tiger-left
        [1 / 3, 1 / 3, 1 / 3], # tiger-right
        [1, 0, 0], # terminal
    ])

    pi_params = reverse_softmax(pi_phi)
    return pi_params

def tiger_alt_start_cam(**kwargs) -> jnp.ndarray:
    pi_phi = jnp.array([
        [0.8, 0.15, 0.05], # init
        [0.2, 0.7, 0.1], # tiger-left
        [0.6, 0.1, 0.3], # tiger-right
        [1, 0, 0], # terminal
    ])
    pi_params = reverse_softmax(pi_phi)
    return pi_params

def get_start_pi(pi_name: str, pi_phi: jnp.ndarray = None, **kwargs):
    if pi_phi is not None:
        return reverse_softmax(pi_phi)

    try:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        pi_params = globals()[pi_name](**kwargs)

    except KeyError as _:
        raise KeyError(f"No policy of the name {pi_name} found in policy_lib")

    return pi_params
