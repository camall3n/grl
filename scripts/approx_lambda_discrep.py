
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.mdp import MDP


def B(mdp: MDP, pi: jnp.ndarray, vp: jnp.ndarray):
    """
    Bellman operator.
    """
    # we repeat values over S x A
    repeated_vp = vp[None, ...]
    repeated_vp = repeated_vp.repeat(mdp.R.shape[0] * mdp.R.shape[1], axis=0)
    repeated_vp_reshaped = repeated_vp.reshape(mdp.R.shape[0], mdp.R.shape[1], -1)

    # g = r + gamma * v(s')
    g = mdp.R + mdp.gamma * repeated_vp_reshaped

    # g * p(s' | s, a)
    new_q = (mdp.T * g).sum(axis=-1)

    # sum_a pi * Q
    new_v = (new_q * pi.T).sum(axis=0)
    return new_v


if __name__ == "__main__":
    spec_name = 'tmaze_5_two_thirds_up'

    pomdp, pi_dict = load_pomdp(spec_name,
                                corridor_length=5,
                                discount=0.9,
                                epsilon=0.1)
    phi_t_phi = pomdp.phi.T @ pomdp.phi
    phi_t_phi_inv_phi = jnp.linalg.inv(phi_t_phi) @ pomdp.phi.T
    proj = pomdp.phi @ phi_t_phi_inv_phi

    pi_phi = pi_dict['Pi_phi'][0]
    pi_s = pomdp.phi @ pi_phi

    def mc_projection(v: jnp.ndarray, steps: int):
        o = v
        for i in range(steps):
            o = B(pomdp, pi_s, o)
        return proj @ o

    def td_projection(v: jnp.ndarray, steps: int):
        o = v
        for i in range(steps):
            o = proj @ B(pomdp, pi_s, o)
        return o

    val_func = jnp.zeros(pomdp.state_space.n)

    for k in [0, 1, 2, 3, 4, 5, 10, 20]:
        mc_val = mc_projection(val_func, k)
        td_val = td_projection(val_func, k)

        diff = mc_val - td_val
        mse = (diff ** 2).mean()

        print(f"Difference for {k} steps: {diff}")
        print(f"MSE for {k} steps: {mse}")


