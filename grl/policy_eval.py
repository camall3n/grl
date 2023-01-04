import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, nn
from functools import partial

from .mdp import AbstractMDP
from grl.utils.pe import functional_solve_mdp, functional_solve_amdp, functional_get_occupancy
from grl.utils.loss import *

def lambda_discrep_measures(amdp: AbstractMDP, pi: jnp.ndarray):
    amdp_pe = PolicyEval(amdp)
    state_vals, mc_vals, td_vals = amdp_pe.run(pi)
    pi_occupancy = amdp_pe.get_occupancy(pi)
    pr_oa = (pi_occupancy @ amdp.phi * pi.T)
    discrep = {
        'v': (mc_vals['v'] - td_vals['v'])**2,
        'q': (mc_vals['q'] - td_vals['q'])**2,
        'mc_vals_q': mc_vals['q'],
        'td_vals_q': td_vals['q'],
        'mc_vals_v': mc_vals['v'],
        'td_vals_v': td_vals['v'],
        'state_vals_v': state_vals['v'],
        'state_vals_q': state_vals['q'],
        'p0': amdp.p0.copy()
    }
    discrep['q_sum'] = (discrep['q'] * pr_oa).sum()
    return discrep


class PolicyEval:
    def __init__(self, amdp, verbose=True, error_type: str = 'l2', value_type: str = 'q'):
        """
        :param amdp:     AMDP
        :param verbose:  log everything
        """
        self.amdp = amdp
        self.verbose = verbose
        self.error_type = error_type
        self.value_type = value_type

        # memory
        if self.value_type == 'v':
            if self.error_type == 'l2':
                self.fn_mem_loss = mem_v_l2_loss
            elif self.error_type == 'abs':
                self.fn_mem_loss = mem_v_abs_loss
        elif self.value_type == 'q':
            if self.error_type == 'l2':
                self.fn_mem_loss = mem_q_l2_loss
            elif self.error_type == 'abs':
                self.fn_mem_loss = mem_q_abs_loss

        # policy
        self.functional_loss_fn = self.functional_mse_loss
        self.loss_fn = self.mse_loss
        if self.error_type == 'max':
            self.functional_loss_fn = self.functional_max_loss
            self.loss_fn = self.max_loss
        elif self.error_type == 'abs':
            self.functional_loss_fn = self.functional_abs_loss
            self.loss_fn = self.abs_loss

    def run(self, pi_abs):
        """
        :param pi_abs: policy to evaluate, defined over abstract state space
        """
        return analytical_pe(pi_abs, self.amdp.phi, self.amdp.T, self.amdp.R, self.amdp.p0,
                             self.amdp.gamma)

    def _solve_mdp(self, mdp, pi):
        v, q = functional_solve_mdp(pi, mdp.T, mdp.R, mdp.gamma)

        return {'v': v, 'q': q}

    def get_occupancy(self, pi: jnp.ndarray):
        """
        Finds the visitation count, C_pi(s), of each state.
        For all s, C_pi(s) = p0(s) + sum_s^ sum_a[C_pi(s^) * gamma * T(s|a,s^) * pi(a|s^)],
          where s^ is the prev state
        """
        pi_ground = self.amdp.phi @ pi
        return functional_get_occupancy(pi_ground, self.amdp.T, self.amdp.p0, self.amdp.gamma)

    def _solve_amdp(self, mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                    pi: jnp.ndarray):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        return functional_solve_amdp(mdp_q_vals, p_pi_of_s_given_o, pi)

    ##########
    # Helpers for gradient/heatmap stuff
    ##########

    def mse_loss(self, pi, value_type, **kwargs):
        """
        sum_o [V_td^pi(o) - V_mc^pi(o)]^2
        """
        _, mc_vals, td_vals = self.run(pi)
        diff = mc_vals[value_type] - td_vals[value_type]
        return (diff**2).mean()

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_abs_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma)
        diff = mc_vals[value_type] - td_vals[value_type]
        return jnp.abs(diff).mean()

    def abs_loss(self, pi, value_type, **kwargs):
        """
        sum_o |V_td^pi(o) - V_mc^pi(o)|
        """
        _, mc_vals, td_vals = self.run(pi)
        diff = mc_vals[value_type] - td_vals[value_type]
        return jnp.abs(diff).mean()

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_mse_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma)
        diff = mc_vals[value_type] - td_vals[value_type]
        return (diff**2).mean()

    def max_loss(self, pi, value_type, **kwargs):
        """
        max_o abs[V_td^pi(o) - V_mc^pi(o)]
        """
        _, mc_vals, td_vals = self.run(pi)
        return np.abs(mc_vals[value_type] - td_vals[value_type]).max()

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_max_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma)
        return jnp.abs(mc_vals[value_type] - td_vals[value_type]).max()

    def memory_loss(self, mem_params, value_type, **kwargs):
        amdp = memory_cross_product(self.amdp, mem_params)
        pe = PolicyEval(amdp, verbose=False)
        return pe.mse_loss(kwargs['pi_abs'], value_type)

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
        return self.functional_memory_update(mem_params, self.amdp.gamma, lr, pi, self.amdp.T,
                                             self.amdp.R, self.amdp.phi, self.amdp.p0)

    @partial(jit, static_argnames=['self', 'gamma', 'lr'])
    def functional_memory_update(self, params: jnp.ndarray, gamma: float, lr: float,
                                 pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                                 p0: jnp.ndarray):
        loss, params_grad = value_and_grad(self.fn_mem_loss, argnums=0)(params, gamma, pi, T, R,
                                                                        phi, p0)
        params -= lr * params_grad

        return loss, params
