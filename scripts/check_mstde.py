import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
from jax_tqdm import scan_tqdm

import optax

from grl.utils.loss import pg_objective_func, mstd_err
from grl.utils.optimizer import get_optimizer
from grl.environment import load_pomdp


if __name__ == "__main__":
    spec = "tmaze_5_two_thirds_up"
    steps = 5000
    lr = 0.01
    seed = 2020
    residual = False

    rng = random.PRNGKey(seed=seed)
    pomdp, pi_dict = load_pomdp(spec)

    rng, pi_rng = random.split(rng)
    pi_shape = (pomdp.observation_space.n, pomdp.action_space.n)
    pi_params = random.normal(pi_rng, shape=pi_shape) * 0.5

    optim = get_optimizer("adam", lr)

    pi_tx_params = optim.init(pi_params)

    def update_pg_step(params, tx_params, pomdp):
        outs, params_grad = value_and_grad(pg_objective_func, has_aux=True)(params, pomdp)
        v_0, (td_v_vals, td_q_vals) = outs

        # We add a negative here to params_grad b/c we're trying to
        # maximize the PG objective (value of start state).
        params_grad = -params_grad
        updates, tx_params = optim.update(params_grad, tx_params, params)
        params = optax.apply_updates(params, updates)
        outs = (params, tx_params, pomdp)
        return outs, {'v0': v_0, 'v': td_v_vals, 'q': td_q_vals}

    @jit
    @scan_tqdm(steps)
    def policy_improvement_scan_wrapper(inps, i):
        params, tx_params, pomdp = inps
        outs, info = update_pg_step(params, tx_params, pomdp)

        params, tx_params, pomdp = outs
        mstde, _, _ = mstd_err(params, pomdp, residual=residual)

        info['mstde'] = mstde
        return outs, info

    outs, info = jax.lax.scan(policy_improvement_scan_wrapper, (pi_params, pi_tx_params, pomdp), jnp.arange(steps))

    print()
