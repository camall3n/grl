import jax.numpy as jnp
from jax import jit, value_and_grad, nn
from .pe import analytical_pe
from grl.memory import functional_memory_cross_product, memory_cross_product

def mem_diff(value_type: str, mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray,
             T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    _, mc_vals, td_vals = analytical_pe(pi, phi_x, T_x, R_x, p0_x, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return diff, mc_vals, td_vals

def mem_v_l2_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                  R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('v', mem_params, gamma, pi, T, R, phi, p0)
    return (diff**2).mean()

def mem_q_l2_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                  R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('q', mem_params, gamma, pi, T, R, phi, p0)
    diff = diff * pi.T
    return (diff**2).mean()

def mem_v_abs_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                   R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('v', mem_params, gamma, pi, T, R, phi, p0)
    return jnp.abs(diff).mean()

def mem_q_abs_loss(mem_params: jnp.ndarray, gamma: float, pi: jnp.ndarray, T: jnp.ndarray,
                   R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    diff, _, _ = mem_diff('q', mem_params, gamma, pi, T, R, phi, p0)
    diff = diff * pi.T
    return jnp.abs(diff).mean()

def pi_calc_diff(value_type: str, pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray,
              R: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray):
    pi = nn.softmax(pi_params, axis=-1)
    _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return diff, mc_vals, td_vals, pi

def pi_discrep_v_l2_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                         phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, _ = pi_calc_diff('v', pi_params, gamma, T, R, phi, p0)
    return (diff**2).mean(), (mc_vals, td_vals)

def pi_discrep_q_l2_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                         phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, pi = pi_calc_diff('q', pi_params, gamma, T, R, phi, p0)
    diff = diff * pi.T
    return (diff**2).mean(), (mc_vals, td_vals)

def pi_discrep_v_abs_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                          phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, _ = pi_calc_diff('v', pi_params, gamma, T, R, phi, p0)
    return jnp.abs(diff).mean(), (mc_vals, td_vals)

def pi_discrep_q_abs_loss(pi_params: jnp.ndarray, gamma: float, T: jnp.ndarray, R: jnp.ndarray,
                          phi: jnp.ndarray, p0: jnp.ndarray):
    diff, mc_vals, td_vals, pi = pi_calc_diff('q', pi_params, gamma, T, R, phi, p0)
    diff = diff * pi.T
    return jnp.abs(diff).mean(), (mc_vals, td_vals)
