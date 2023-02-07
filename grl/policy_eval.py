import jax.numpy as jnp
from jax import value_and_grad, jit, nn
from functools import partial

from grl.utils.policy_eval import analytical_pe
from grl.utils.loss import mem_discrep_loss, mem_abs_td_loss

class PolicyEval:
    def __init__(self, amdp, verbose: bool = True,
                 discrep_type: str = 'ground_truth',
                 error_type: str = 'l2', value_type: str = 'q',
                 weight_discrep: bool = False):
        """
        :param amdp:     AMDP
        :param verbose:  log everything
        """
        self.amdp = amdp
        self.verbose = verbose
        self.discrep_type = discrep_type
        self.error_type = error_type
        self.value_type = value_type
        self.weight_discrep = weight_discrep

        partial_mem_discrep_loss = partial(mem_discrep_loss,
                                           value_type=self.value_type,
                                           error_type=self.error_type,
                                           weight_discrep=self.weight_discrep)
        if self.discrep_type == 'abs_td':
            partial_mem_discrep_loss = partial(mem_abs_td_loss,
                                               value_type=self.value_type,
                                               error_type=self.error_type,
                                               weight_discrep=self.weight_discrep)
        self.fn_mem_loss = jit(partial_mem_discrep_loss, static_argnames=['gamma'])

        # policy
        self.functional_loss_fn = self.functional_mse_loss
        if self.error_type == 'max':
            self.functional_loss_fn = self.functional_max_loss
        elif self.error_type == 'abs':
            self.functional_loss_fn = self.functional_abs_loss

    def run(self, pi_abs):
        """
        :param pi_abs: policy to evaluate, defined over abstract state space
        """
        return analytical_pe(pi_abs, self.amdp.phi, self.amdp.T, self.amdp.R, self.amdp.p0,
                             self.amdp.gamma)

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_abs_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals, _ = analytical_pe(pi, phi, T, R, p0, gamma)
        diff = mc_vals[value_type] - td_vals[value_type]
        return jnp.abs(diff).mean()

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_mse_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals, _ = analytical_pe(pi, phi, T, R, p0, gamma)
        diff = mc_vals[value_type] - td_vals[value_type]
        return (diff**2).mean()

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_max_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals, _ = analytical_pe(pi, phi, T, R, p0, gamma)
        return jnp.abs(mc_vals[value_type] - td_vals[value_type]).max()

    def policy_update(self, params: jnp.ndarray, value_type: str, lr: float, *args, **kwargs):
        return self.functional_policy_update(params, value_type, self.amdp.gamma, lr, self.amdp.T,
                                             self.amdp.R, self.amdp.phi, self.amdp.p0)

    @partial(jit, static_argnames=['self', 'gamma', 'value_type', 'lr'])
    def functional_policy_update(self, params: jnp.ndarray, value_type: str, gamma: float,
                                 lr: float, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                                 p0: jnp.ndarray):
        pi = nn.softmax(params, axis=-1)
        loss, params_grad = value_and_grad(self.functional_loss_fn, argnums=0)(pi, value_type, phi,
                                                                               T, R, p0, gamma)
        params -= lr * params_grad
        return loss, params

    def memory_update(self, mem_params: jnp.ndarray, value_type: str, lr: float, pi: jnp.ndarray):
        assert value_type == self.value_type
        return self.functional_memory_update(mem_params, self.amdp.gamma, lr, pi, self.amdp.phi, self.amdp.T,
                                             self.amdp.R, self.amdp.p0)

    @partial(jit, static_argnames=['self', 'gamma', 'lr'])
    def functional_memory_update(self, params: jnp.ndarray, gamma: float, lr: float,
                                 pi: jnp.ndarray, phi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray,
                                 p0: jnp.ndarray):
        loss, params_grad = value_and_grad(self.fn_mem_loss, argnums=0)(params, gamma, pi,
                                                                        phi, T, R, p0)
        params -= lr * params_grad

        return loss, params
