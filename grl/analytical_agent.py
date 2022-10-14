import jax.numpy as jnp
from jax import jit, value_and_grad, random
from jax.nn import softmax
from functools import partial
from typing import Sequence

from grl.policy_eval import functional_get_occupancy, get_p_s_given_o, functional_solve_mdp, functional_create_td_model
from grl.policy_eval import functional_memory_cross_product, analytical_pe
from grl.mdp import AbstractMDP
from grl.utils import golrot_init

def memory_loss(mem_params: jnp.ndarray, gamma: float, value_type: str,
                           pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                           p0: jnp.ndarray):
    T_mem = softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    _, mc_vals, td_vals = analytical_pe(pi, phi_x, T_x, R_x, p0_x, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return (diff ** 2).mean()

def pg_objective_func(pi_params: jnp.ndarray, gamma: float,
                      T: jnp.ndarray, phi: jnp.ndarray, p0: jnp.ndarray,
                      R: jnp.ndarray):
    """
    Policy gradient objective function:
    \sum_{s_0} p(s_0) v_pi(s_0)
    """
    pi_abs = softmax(pi_params, axis=-1)
    pi_ground = phi @ pi_abs
    occupancy = functional_get_occupancy(pi_ground, T, p0, gamma)

    p_pi_of_s_given_o = get_p_s_given_o(phi, occupancy)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, phi, T, R)
    td_v_vals, _ = functional_solve_mdp(pi_abs, T_obs_obs, R_obs_obs, gamma)
    p_init_obs = p0 @ phi
    return jnp.dot(p_init_obs, td_v_vals)

class AnalyticalAgent:
    """
    Analytical agent that learns optimal policy params based on an
    analytic policy gradient.
    """
    def __init__(self,
                 pi_params: jnp.ndarray, mem_params: jnp.ndarray = None,
                 discrep_type: str = 'q', rand_key: random.PRNGKey = random.PRNGKey(2022),
                 pi_softmax_temp: float = 1):
        self.pi_params = pi_params
        self.og_n_obs = self.pi_params.shape[0]
        self.mem_params = mem_params
        self.discrep_type = discrep_type
        self.policy_objective_func = jit(pg_objective_func, static_argnames='gamma')
        self.memory_objective_func = jit(memory_loss, static_argnames=['gamma', 'value_type'])
        self.pi_softmax_temp = pi_softmax_temp

        self.rand_key = rand_key

    def reset_pi_params(self, pi_shape: Sequence[int] = None):
        self.rand_key, pi_reset_key = random.split(self.rand_key)

        if pi_shape is None:
            pi_shape = self.pi_params.shape
        self.pi_params = golrot_init(pi_shape)

    def repeat_pi_over_mem(self):
        if self.pi_params.shape[0] != self.og_n_obs:
            raise NotImplementedError("Have not implemented adding bits to already existing memory.")

        add_n_mem_states = self.mem_params.shape[-1]

        self.pi_params = self.pi_params.repeat(add_n_mem_states, axis=0)

    @partial(jit, static_argnames=['self', 'gamma', 'lr'])
    def functional_policy_update(self, params: jnp.ndarray, gamma: float, lr: float,
                                 T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                                 p0: jnp.ndarray):
        v_0, params_grad = value_and_grad(self.policy_objective_func, argnums=0)(params, gamma, T, phi, p0, R)
        params += lr * params_grad
        return v_0, params

    def policy_improvement(self, amdp: AbstractMDP, lr: float):
        v_0, new_pi_params = self.functional_policy_update(self.pi_params, amdp.gamma, lr,
                                                       amdp.T, amdp.R, amdp.phi, amdp.p0)
        self.pi_params = new_pi_params
        return v_0

    @partial(jit, static_argnames=['self', 'gamma', 'value_type', 'lr'])
    def functional_memory_update(self, params: jnp.ndarray, value_type: str, gamma: float,
                                 lr: float, pi_params: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray,
                                 phi: jnp.ndarray, p0: jnp.ndarray):
        pi = softmax(pi_params / self.pi_softmax_temp, axis=-1)
        loss, params_grad = value_and_grad(self.memory_objective_func,
                                           argnums=0)(params, gamma, value_type, pi, T, R, phi, p0)
        params -= lr * params_grad

        return loss, params

    def memory_improvement(self, amdp: AbstractMDP, lr: float):
        assert self.mem_params is not None, 'I have no memory params'
        loss, new_mem_params = self.functional_memory_update(self.mem_params, self.discrep_type,
                                                             amdp.gamma, lr, self.pi_params,
                                                             amdp.T, amdp.R, amdp.phi, amdp.p0)
        self.mem_params = new_mem_params
        return loss

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()

        # delete unpickleable jitted functions
        del state['policy_objective_func']
        del state['memory_objective_func']
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

        # restore jitted functions
        self.policy_objective_func = jit(pg_objective_func, static_argnames='gamma')
        self.memory_objective_func = jit(memory_loss, static_argnames=['gamma', 'value_type'])

