import jax.numpy as jnp
from jax import value_and_grad, jit, nn
from functools import partial

from grl.utils.policy_eval import analytical_pe
from grl.utils.loss import mem_discrep_loss, mem_magnitude_td_loss, policy_discrep_loss
from grl.mdp import AbstractMDP

class PolicyEval:
    def __init__(self, amdp: AbstractMDP, verbose: bool = True,
                 discrep_type: str = 'ground_truth',
                 error_type: str = 'l2', value_type: str = 'q',
                 alpha: float = 1):
        """
        :param amdp:     AMDP
        :param verbose:  log everything
        """
        self.amdp = amdp
        self.verbose = verbose
        self.discrep_type = discrep_type
        self.error_type = error_type
        self.value_type = value_type
        self.alpha = alpha

        partial_mem_discrep_loss = partial(mem_discrep_loss,
                                           value_type=self.value_type,
                                           error_type=self.error_type,
                                           alpha=self.alpha)
        if self.discrep_type == 'abs_td':
            partial_mem_discrep_loss = partial(mem_magnitude_td_loss,
                                               value_type=self.value_type,
                                               error_type=self.error_type,
                                               alpha=self.alpha)
        self.fn_mem_loss = jit(partial_mem_discrep_loss)

        partial_policy_discrep_loss = partial(policy_discrep_loss,
                                              value_type=self.value_type,
                                              error_type=self.error_type,
                                              alpha=self.alpha)
        self.policy_discrep_objective_func = jit(partial_policy_discrep_loss)

    def run(self, pi_abs):
        """
        :param pi_abs: policy to evaluate, defined over abstract state space
        """
        return analytical_pe(pi_abs, self.amdp)

    def policy_update(self, params: jnp.ndarray, lr: float, *args, **kwargs):
        return self.functional_policy_update(params, lr, self.amdp)

    @partial(jit, static_argnames=['self', 'lr'])
    def functional_policy_update(self, params: jnp.ndarray, lr: float, amdp: AbstractMDP):
        outs, params_grad = value_and_grad(self.policy_discrep_objective_func, has_aux=True, argnums=0)(params, amdp)
        loss, _ = outs
        params -= lr * params_grad
        return loss, params

    def memory_update(self, mem_params: jnp.ndarray, lr: float, pi: jnp.ndarray):
        return self.functional_memory_update(mem_params, lr, pi, self.amdp)

    @partial(jit, static_argnames=['self', 'lr'])
    def functional_memory_update(self, params: jnp.ndarray, lr: float,
                                 pi: jnp.ndarray, amdp: AbstractMDP):
        loss, params_grad = value_and_grad(self.fn_mem_loss, argnums=0)(params, pi, amdp)
        params -= lr * params_grad

        return loss, params
