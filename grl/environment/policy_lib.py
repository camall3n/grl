import jax.numpy as jnp

from grl.utils.math import reverse_softmax

def tiger_alt_start_pi(**kwargs) -> jnp.ndarray:
    # actions are listen, open-left, open-right
    pi_phi = jnp.array([
        [1, 0, 0],  # init
        [1, 0, 0],  # tiger-left
        [1, 0, 0],  # tiger-right
        [1, 0, 0],  # terminal
    ])

    pi_params = reverse_softmax(pi_phi)
    return pi_params

def get_start_pi(pi_name: str, **kwargs):
    try:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        pi_params = locals()[pi_name](**kwargs)

    except AttributeError as _:
        raise AttributeError(f"No policy of the name {pi_name} found in policy_lib")
