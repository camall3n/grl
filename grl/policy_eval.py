import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, nn
from functools import partial

from .mdp import MDP, AbstractMDP
from .memory import functional_memory_cross_product, memory_cross_product

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

@partial(jit, static_argnames=['gamma'])
def functional_get_occupancy(pi_ground: jnp.ndarray, T: jnp.ndarray, p0: jnp.ndarray,
                             gamma: float):
    Pi_pi = pi_ground.transpose()[..., None]
    T_pi = (Pi_pi * T).sum(axis=0)  # T^π(s'|s)

    # A*C_pi(s) = b
    # A = (I - \gamma (T^π)^T)
    # b = P_0
    A = jnp.eye(T.shape[-1]) - gamma * T_pi.transpose()
    b = p0
    return jnp.linalg.solve(A, b)

@jit
def get_p_s_given_o(phi: jnp.ndarray, occupancy: jnp.ndarray):
    repeat_occupancy = jnp.repeat(occupancy[..., None], phi.shape[-1], -1)

    # Q vals
    p_of_o_given_s = phi.astype('float')
    w = repeat_occupancy * p_of_o_given_s

    p_pi_of_s_given_o = w / (w.sum(axis=0) + 1e-10)
    return p_pi_of_s_given_o

@partial(jit, static_argnames=['gamma'])
def functional_solve_mdp(pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, gamma: float):
    """
    Solves for V using linear equations.
    For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
    """
    Pi_pi = pi.transpose()[..., None]
    T_pi = (Pi_pi * T).sum(axis=0)  # T^π(s'|s)
    R_pi = (Pi_pi * T * R).sum(axis=0).sum(axis=-1)  # R^π(s)

    # A*V_pi(s) = b
    # A = (I - \gamma (T^π))
    # b = R^π
    A = (jnp.eye(T.shape[-1]) - gamma * T_pi)
    b = R_pi
    v_vals = jnp.linalg.solve(A, b)

    R_sa = (T * R).sum(axis=-1)  # R(s,a)
    q_vals = (R_sa + (gamma * T @ v_vals))

    return v_vals, q_vals

def mem_diff(value_type: str, mem_params: jnp.ndarray, gamma: float,
             pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
             p0: jnp.ndarray):
    T_mem = nn.softmax(mem_params, axis=-1)
    T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
    _, mc_vals, td_vals = analytical_pe(pi, phi_x, T_x, R_x, p0_x, gamma)
    diff = mc_vals[value_type] - td_vals[value_type]
    return diff, mc_vals, td_vals

def mem_v_l2_loss(mem_params: jnp.ndarray, gamma: float,
                pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                p0: jnp.ndarray):
    diff, _, _ = mem_diff('v', mem_params, gamma, pi, T, R, phi, p0)
    return (diff ** 2).mean()

def mem_q_l2_loss(mem_params: jnp.ndarray, gamma: float,
                  pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                  p0: jnp.ndarray):
    diff, _, _ = mem_diff('q', mem_params, gamma, pi, T, R, phi, p0)
    diff = diff * pi.T
    return (diff ** 2).mean()

def mem_v_abs_loss(mem_params: jnp.ndarray, gamma: float,
                     pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                     p0: jnp.ndarray):
    diff, _, _ = mem_diff('v', mem_params, gamma, pi, T, R, phi, p0)
    return jnp.abs(diff).mean()

def mem_q_abs_loss(mem_params: jnp.ndarray, gamma: float,
                     pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                     p0: jnp.ndarray):
    diff, _, _ = mem_diff('q', mem_params, gamma, pi, T, R, phi, p0)
    diff = diff * pi.T
    return jnp.abs(diff).mean()

@jit
def functional_solve_amdp(mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                          pi_abs: jnp.ndarray):
    # Q vals
    amdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

    # V vals
    amdp_v_vals = (amdp_q_vals * pi_abs.T).sum(0)

    return {'v': amdp_v_vals, 'q': amdp_q_vals}

@jit
def functional_create_td_model(p_pi_of_s_given_o: jnp.ndarray, phi: jnp.ndarray,
                               T: jnp.ndarray, R: jnp.ndarray):
    # creates an (n_obs * n_obs) x 2 array of all possible observation to observation pairs.
    # we flip here so that we have curr_obs, next_obs (order matters).
    obs_idx_product = jnp.flip(
        jnp.dstack(jnp.meshgrid(jnp.arange(phi.shape[-1]), jnp.arange(phi.shape[-1]))).reshape(-1, 2), -1)

    # this gives us (n_obs * n_obs) x states x 1 and (n_obs * n_obs) x 1 x states
    curr_s_given_o = p_pi_of_s_given_o[:, obs_idx_product[:, 0]].T[..., None]
    next_o_given_s = jnp.expand_dims(phi[:, obs_idx_product[:, 1]].T, 1)

    # outer product here
    o_to_next_o = jnp.expand_dims(curr_s_given_o * next_o_given_s, 1)

    # This is p(o, s, a, s', o')
    # the probability that o goes to o', via each path (s, a) -> s'.
    # Shape is (n_obs * n_obs) x |A| x |S| x |S|
    T_contributions = T * o_to_next_o

    # |A| x (n_obs * n_obs)
    T_obs_obs_flat = T_contributions.sum(-1).sum(-1).T

    # |A| x n_obs x n_obs
    T_obs_obs = T_obs_obs_flat.reshape(T.shape[0], phi.shape[-1], phi.shape[-1])

    # You want everything to sum to one
    denom = T_obs_obs_flat.T[..., None, None]
    denom_no_zero = denom + (denom == 0).astype(denom.dtype)

    R_contributions = (R * T_contributions) / denom_no_zero
    R_obs_obs_flat = R_contributions.sum(-1).sum(-1).T
    R_obs_obs = R_obs_obs_flat.reshape(R.shape[0], phi.shape[-1], phi.shape[-1])

    return T_obs_obs, R_obs_obs

@partial(jit, static_argnames=['gamma'])
def analytical_pe(pi_obs: jnp.ndarray, phi: jnp.ndarray, T: jnp.ndarray,
                  R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
    # observation policy, but expanded over states
    pi_state = phi @ pi_obs

    # MC*
    state_v, state_q = functional_solve_mdp(pi_state, T, R, gamma)
    state_vals = {'v': state_v, 'q': state_q}

    occupancy = functional_get_occupancy(pi_state, T, p0, gamma)

    p_pi_of_s_given_o = get_p_s_given_o(phi, occupancy)
    mc_vals = functional_solve_amdp(state_q, p_pi_of_s_given_o, pi_obs)

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, phi, T, R)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_obs, T_obs_obs, R_obs_obs, gamma)
    td_vals = {'v': td_v_vals, 'q': td_q_vals}

    return state_vals, mc_vals, td_vals

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

    def _create_td_model(self, p_pi_of_s_given_o: jnp.ndarray):
        T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, self.amdp.phi,
                                                          self.amdp.T, self.amdp.R)
        return MDP(T_obs_obs, R_obs_obs, self.amdp.p0, self.amdp.gamma)

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
        loss, params_grad = value_and_grad(self.functional_loss_fn,
                                           argnums=0)(pi, value_type, phi, T, R, p0, gamma)
        params -= lr * params_grad
        return loss, params

    def memory_update(self, mem_params: jnp.ndarray, value_type: str, lr: float, pi: jnp.ndarray):
        assert value_type == self.value_type
        return self.functional_memory_update(mem_params, self.amdp.gamma, lr, pi,
                                             self.amdp.T, self.amdp.R, self.amdp.phi, self.amdp.p0)

    @partial(jit, static_argnames=['self', 'gamma', 'lr'])
    def functional_memory_update(self, params: jnp.ndarray, gamma: float,
                                 lr: float, pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray,
                                 phi: jnp.ndarray, p0: jnp.ndarray):
        loss, params_grad = value_and_grad(self.fn_mem_loss,
                                           argnums=0)(params, gamma, pi, T, R, phi, p0)
        params -= lr * params_grad

        return loss, params

