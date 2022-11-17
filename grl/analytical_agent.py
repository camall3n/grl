import jax.numpy as jnp
from jax import jit, value_and_grad, random
from jax.nn import softmax
from functools import partial
from typing import Sequence

from grl.policy_eval import functional_get_occupancy, get_p_s_given_o, functional_solve_mdp, functional_create_td_model
from grl.policy_eval import functional_memory_cross_product, analytical_pe, memory_loss
from grl.mdp import AbstractMDP
from grl.utils import glorot_init
from grl.vi import policy_iteration_step

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
    td_v_vals, td_q_vals = functional_solve_mdp(pi_abs, T_obs_obs, R_obs_obs, gamma)
    p_init_obs = p0 @ phi
    return jnp.dot(p_init_obs, td_v_vals), (td_v_vals, td_q_vals)

def pi_discrep_loss(pi_params: jnp.ndarray, gamma: float, value_type: str,
                   T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                   p0: jnp.ndarray):
    pi = softmax(pi_params, axis=-1)
    _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return (diff ** 2).mean(), (mc_vals, td_vals)

class AnalyticalAgent:
    """
    Analytical agent that learns optimal policy params based on an
    analytic policy gradient.
    """
    def __init__(self,
                 pi_params: jnp.ndarray, mem_params: jnp.ndarray = None,
                 discrep_type: str = 'v', rand_key: random.PRNGKey = random.PRNGKey(2022),
                 pi_softmax_temp: float = 1, policy_optim_alg: str = 'pi',
                 new_mem_pi: str = 'copy',
                 epsilon: float = 0.1):
        """
        :param policy_optim_alg: What type of policy optimization do we do? (pi | pg)
        """
        self.policy_optim_alg = policy_optim_alg
        self.pi_params = pi_params
        self.og_n_obs = self.pi_params.shape[0]

        self.pg_objective_func = jit(pg_objective_func, static_argnames='gamma')

        self.policy_iteration_update = jit(policy_iteration_step, static_argnames=['gamma', 'eps'])
        self.epsilon = epsilon

        self.policy_discrep_objective_func = jit(pi_discrep_loss, static_argnames=['gamma', 'value_type'])

        self.mem_params = mem_params
        self.new_mem_pi = new_mem_pi
        self.discrep_type = discrep_type
        self.memory_objective_func = jit(memory_loss, static_argnames=['gamma', 'value_type'])
        self.pi_softmax_temp = pi_softmax_temp

        self.rand_key = rand_key

    @property
    def policy(self) -> jnp.ndarray:
        # return the learnt policy
        return softmax(self.pi_params, axis=-1)

    @property
    def memory(self) -> jnp.ndarray:
        return softmax(self.mem_params, axis=-1)

    def reset_pi_params(self, pi_shape: Sequence[int] = None):
        self.rand_key, pi_reset_key = random.split(self.rand_key)

        if pi_shape is None:
            pi_shape = self.pi_params.shape
        self.pi_params = glorot_init(pi_shape)

    def new_pi_over_mem(self):
        if self.pi_params.shape[0] != self.og_n_obs:
            raise NotImplementedError("Have not implemented adding bits to already existing memory.")

        add_n_mem_states = self.mem_params.shape[-1]
        old_pi_params_shape = self.pi_params.shape

        self.pi_params = self.pi_params.repeat(add_n_mem_states, axis=0)

        if self.new_mem_pi == 'random':
            # randomly init policy for new memory state
            new_mem_params = glorot_init(old_pi_params_shape)
            self.pi_params = self.pi_params.at[1::2].set(new_mem_params)

    @partial(jit, static_argnames=['self', 'gamma', 'lr'])
    def functional_pg_update(self, params: jnp.ndarray, gamma: float, lr: float,
                             T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                             p0: jnp.ndarray):
        outs, params_grad = value_and_grad(self.pg_objective_func, has_aux=True)(params, gamma, T, phi, p0, R)
        v_0, (td_v_vals, td_q_vals) = outs
        params += lr * params_grad
        return v_0, td_v_vals, td_q_vals, params

    @partial(jit, static_argnames=['self', 'value_type', 'gamma', 'lr'])
    def functional_dm_update(self, params: jnp.ndarray, gamma: float,
                             value_type: str, lr: float,
                             T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                             p0: jnp.ndarray):
        outs, params_grad = value_and_grad(self.policy_discrep_objective_func, has_aux=True)(params, gamma, value_type,
                                                                                             T, R, phi, p0)
        loss, (mc_vals, td_vals) = outs
        params += lr * params_grad
        return loss, mc_vals, td_vals, params

    def policy_improvement(self, amdp: AbstractMDP, lr: float = None):
        if self.policy_optim_alg == 'pg':
            v_0, prev_td_v_vals, prev_td_q_vals, new_pi_params = self.functional_pg_update(self.pi_params, amdp.gamma, lr,
                                                           amdp.T, amdp.R, amdp.phi, amdp.p0)
            output = {'v_0': v_0, 'prev_td_q_vals': prev_td_q_vals, 'prev_td_v_vals': prev_td_v_vals}
        elif self.policy_optim_alg == 'pi':
            new_pi_params, prev_td_v_vals, prev_td_q_vals = self.policy_iteration_update(self.pi_params, amdp.T, amdp.R, amdp.phi,
                                                                               amdp.p0, amdp.gamma, eps=self.epsilon)
            output = {'prev_td_q_vals': prev_td_q_vals, 'prev_td_v_vals': prev_td_v_vals}
        elif self.policy_optim_alg == 'dm':
            loss, mc_vals, td_vals, new_pi_params = self.functional_dm_update(self.pi_params, amdp.gamma,
                                                                              self.discrep_type, lr,
                                                                              amdp.T, amdp.R,
                                                                              amdp.phi, amdp.p0)
            output = {'loss': loss, 'mc_vals': mc_vals, 'td_vals': td_vals}
        else:
            raise NotImplementedError
        self.pi_params = new_pi_params
        return output

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
        del state['pg_objective_func']
        del state['policy_iteration_update']
        del state['policy_discrep_objective_func']
        del state['memory_objective_func']
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

        # restore jitted functions
        self.pg_objective_func = jit(pg_objective_func, static_argnames='gamma')
        self.policy_iteration_update = jit(policy_iteration_step, static_argnames=['gamma', 'eps'])
        self.policy_discrep_objective_func = jit(pi_discrep_loss, static_argnames=['gamma', 'value_type'])

        self.memory_objective_func = jit(memory_loss, static_argnames=['gamma', 'value_type'])
