import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import softmax
from tqdm import trange
from functools import partial
from typing import Callable

from grl.agent.analytical import AnalyticalAgent
from grl.mdp import POMDP
from grl.memory import memory_cross_product
from grl.utils.augment_policy import deconstruct_aug_policy
from grl.utils.math import glorot_init, greedify, reverse_softmax
from grl.utils.lambda_discrep import lambda_discrep_measures
from grl.utils.loss import discrep_loss, pg_objective_func
from grl.vi import td_pe

def run_memory_iteration(pomdp: POMDP,
                         mem_params: jnp.ndarray,
                         policy_optim_alg: str = 'policy_iter',
                         optimizer_str: str = 'sgd',
                         pi_lr: float = 1.,
                         mi_lr: float = 1.,
                         mi_iterations: int = 1,
                         mi_steps: int = 50000,
                         pi_steps: int = 50000,
                         rand_key: jax.random.PRNGKey = None,
                         error_type: str = 'l2',
                         value_type: str = 'q',
                         objective: str = 'discrep',
                         residual: bool = False,
                         lambda_0: float = 0.,
                         lambda_1: float = 1.,
                         alpha: float = 1.,
                         pi_params: jnp.ndarray = None,
                         kitchen_sink_policies: int = 0,
                         epsilon: float = 0.1,
                         flip_count_prob: bool = False):
    """
    Wrapper function for the Memory Iteration algorithm.
    Memory iteration intersperses memory improvement and policy improvement.
    :param pomdp:                POMDP to do memory iteration on.
    :param pi_lr:               Policy improvement learning rate
    :param mi_lr:               Memory improvement learning rate
    :param policy_optim_alg:    Which policy improvement algorithm do we use?
        (discrep_max: discrepancy maximization | discrep_min: discrepancy minimization | policy_iter: policy iteration
        | policy_grad: policy gradient)
    :param mi_iterations:       How many memory iterations do we do?
    :param mi_steps:            Number of memory improvement steps PER memory iteration step.
    :param pi_steps:            Number of policy improvement steps PER memory iteration step.
    :param flip_count_prob:     Do we flip our count probabilities over observations for memory loss?.
    :param lambda_0:            What's our first lambda parameter for lambda discrep?
    :param lambda_1:            What's our second lambda parameter for lambda discrep?
    :param alpha:               How uniform do we want our lambda discrep weighting?
    """
    assert isinstance(pomdp, POMDP) and pomdp.current_state is None, \
        f"POMDP should be stateless and current_state should be None, got {pomdp.current_state} instead"

    # If pi_params is initialized, we start with some given
    # policy params and don't do the first policy improvement step.
    init_pi_improvement = False
    if pi_params is None:
        init_pi_improvement = True
        pi_params = glorot_init((pomdp.observation_space.n, pomdp.action_space.n), scale=0.2)
    initial_policy = softmax(pi_params, axis=-1)

    agent = AnalyticalAgent(pi_params,
                            rand_key,
                            optim_str=optimizer_str,
                            pi_lr=pi_lr,
                            mi_lr=mi_lr,
                            mem_params=mem_params,
                            policy_optim_alg=policy_optim_alg,
                            error_type=error_type,
                            value_type=value_type,
                            objective=objective,
                            residual=residual,
                            lambda_0=lambda_0,
                            lambda_1=lambda_1,
                            alpha=alpha,
                            epsilon=epsilon,
                            flip_count_prob=flip_count_prob)

    discrep_loss_fn = partial(discrep_loss,
                              value_type=value_type,
                              error_type=error_type,
                              alpha=alpha)

    info, agent = memory_iteration(agent,
                                   pomdp,
                                   mi_iterations=mi_iterations,
                                   pi_per_step=pi_steps,
                                   mi_per_step=mi_steps,
                                   init_pi_improvement=init_pi_improvement,
                                   kitchen_sink_policies=kitchen_sink_policies,
                                   discrep_loss=discrep_loss_fn)

    get_measures = partial(lambda_discrep_measures, discrep_loss_fn=discrep_loss_fn)

    info['initial_policy'] = initial_policy
    # we get lambda discrepancies here
    # initial policy lambda-discrepancy
    info['initial_policy_stats'] = get_measures(pomdp, initial_policy)
    info['initial_improvement_stats'] = get_measures(pomdp, info['initial_improvement_policy'])
    greedy_initial_improvement_policy = info['initial_improvement_policy']
    if policy_optim_alg == 'policy_iter':
        greedy_initial_improvement_policy = greedify(info['initial_improvement_policy'])
    info['greedy_initial_improvement_stats'] = get_measures(pomdp,
                                                            greedy_initial_improvement_policy)

    if policy_optim_alg in ['discrep_max', 'discrep_min'] or not init_pi_improvement:
        info['td_optimal_policy_stats'] = get_measures(pomdp, info['td_optimal_memoryless_policy'])
        info['greedy_td_optimal_policy_stats'] = get_measures(
            pomdp, greedify(info['td_optimal_memoryless_policy']))

    # Initial memory pomdp w/ initial improvement policy discrep
    if 'initial_mem_params' in info and info['initial_mem_params'] is not None:
        init_mem_pomdp = memory_cross_product(info['initial_mem_params'], pomdp)
        info['initial_mem_stats'] = get_measures(init_mem_pomdp,
                                                 info['initial_expanded_improvement_policy'])

    # Final memory w/ final policy discrep
    final_mem_pomdp = memory_cross_product(agent.mem_params, pomdp)
    info['final_mem_stats'] = get_measures(final_mem_pomdp, agent.policy)
    greedy_final_policy = agent.policy
    if policy_optim_alg == 'policy_iter':
        greedy_final_policy = greedify(agent.policy)
    info['greedy_final_mem_stats'] = get_measures(final_mem_pomdp, greedy_final_policy)

    def perf_from_stats(stats: dict) -> float:
        return np.dot(stats['state_vals_v'], stats['p0']).item()

    print("Finished Memory Iteration.")
    print(f"Initial policy performance: {perf_from_stats(info['initial_policy_stats']):.4f}")
    print(
        f"Initial improvement performance: {perf_from_stats(info['initial_improvement_stats']):.4f}"
    )
    if 'greedy_td_optimal_policy_stats' in info:
        print(
            f"TD-optimal policy performance: {perf_from_stats(info['greedy_td_optimal_policy_stats']):.4f}"
        )
    print(f"Final performance after MI: {perf_from_stats(info['greedy_final_mem_stats']):.4f}")

    return info, agent

def memory_iteration(
    agent: AnalyticalAgent,
    init_pomdp: POMDP,
    pi_per_step: int = 50000,
    mi_per_step: int = 50000,
    mi_iterations: int = 1,
    init_pi_improvement: bool = True,
    kitchen_sink_policies: int = 0,
    discrep_loss: Callable = None,
    log_every: int = 1000,
):
    """
    The memory iteration algorithm. This algorithm flips between improving
    policy parameters to maximize return, and memory parameters to minimize lambda discrepancy.
    Each step of memory iteration includes multiple steps of memory improvement and policy improvement.
    :param agent:               Agent instance to run memory iteration.
    :param init_pomdp:           Initial abstract MDP for us to train our agent on.
    :param pi_lr:               Policy improvement learning rate
    :param mi_lr:               Memory improvement learning rate
    :param pi_per_step:         Number of policy improvement steps PER memory iteration step.
    :param mi_per_step:         Number of memory improvement steps PER memory iteration step.
    :param mi_iterations:       Number of steps of memory iteration to perform.
    :param init_pi_improvement: Do we start out with an initial policy improvement step?
    :param kitchen_sink_policies: Do we select our initial policy based on randomly selected policies + TD optimal?
    :param log_every:           How often do we log stats?
    """
    info = {'policy_improvement_outputs': [], 'mem_loss': []}
    td_v_vals, td_q_vals = td_pe(agent.policy, init_pomdp)
    info['initial_values'] = {'v': td_v_vals, 'q': td_q_vals}

    if agent.policy_optim_alg in ['discrep_max', 'discrep_min'] or \
            (not init_pi_improvement and agent.policy_optim_alg not in ['policy_grad', 'policy_mem_grad']):
        initial_pi_params = agent.pi_params.copy()

        og_policy_optim_algo = agent.policy_optim_alg

        # Change modes, run policy iteration
        agent.policy_optim_alg = 'policy_iter'
        print(f"Calculating TD-optimal memoryless policy over {pi_per_step} steps")
        pi_improvement(agent, init_pomdp, iterations=pi_per_step, log_every=log_every)
        info['td_optimal_memoryless_policy'] = agent.policy.copy()
        print(f"Converged to TD-optimal memoryless policy: \n{agent.policy}\n")

        # reset our policy again
        agent.pi_params = initial_pi_params
        agent.policy_optim_alg = og_policy_optim_algo

    if init_pi_improvement:
        # initial policy improvement
        poa = agent.policy_optim_alg
        prev_optim = agent.pi_optim_state
        prev_obj_func = agent.pg_objective_func
        if agent.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
            agent.policy_optim_alg = 'policy_grad'
            agent.pg_objective_func = jax.jit(pg_objective_func)
            agent.pi_optim_state = agent.pi_optim.init(agent.pi_params)

        print("Initial policy improvement step")
        initial_outputs = pi_improvement(agent,
                                         init_pomdp,
                                         iterations=pi_per_step,
                                         log_every=log_every)

        info['policy_improvement_outputs'].append(initial_outputs)
        info['initial_improvement_policy'] = agent.policy.copy()
        if poa != agent.policy_optim_alg:
            agent.policy_optim_alg = poa
            agent.pg_objective_func = prev_obj_func
            agent.pi_optim_state = prev_optim

        # here we test random policies
        if kitchen_sink_policies > 0:
            best_lambda_discrep = discrep_loss(agent.policy, init_pomdp)[0].item()
            best_pi_params = agent.pi_params

            print(f"Finding the argmax LD over {kitchen_sink_policies} random policies")
            for i in range(kitchen_sink_policies):
                agent.reset_pi_params()
                lambda_discrep = discrep_loss(agent.policy, init_pomdp)[0].item()
                if lambda_discrep > best_lambda_discrep:
                    best_pi_params = agent.pi_params
                    best_lambda_discrep = lambda_discrep
            agent.pi_params = best_pi_params

        info['starting_policy'] = agent.policy.copy()

    print(f"Starting (unexpanded) policy: \n{agent.policy}\n")

    if agent.mem_params is not None and agent.policy_optim_alg not in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
        # we have to set our policy over our memory MDP now
        # we do so with a random/copied policy given new memory bit
        info['initial_mem_params'] = agent.mem_params

        agent.new_pi_over_mem()
        info['initial_expanded_improvement_policy'] = agent.policy.copy()
        print(f"Starting (expanded) policy: \n{agent.policy}\n")
        print(f"Starting memory: \n{agent.memory}\n")

    pomdp = copy.deepcopy(init_pomdp)

    for mem_it in range(mi_iterations):
        if agent.mem_params is not None and agent.policy_optim_alg not in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
            print(f"Start MI {mem_it}")
            mem_loss = mem_improvement(agent,
                                       init_pomdp,
                                       iterations=mi_per_step,
                                       log_every=log_every)
            info['mem_loss'].append(mem_loss)

            # Plotting for memory iteration
            print(f"Learnt memory for iteration {mem_it}: \n"
                  f"{agent.memory}")

            # Make a NEW memory AMDP
            pomdp = memory_cross_product(agent.mem_params, init_pomdp)

        if pi_per_step > 0:
            if agent.policy_optim_alg not in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
                # reset our policy parameters
                agent.reset_pi_params((pomdp.observation_space.n, pomdp.action_space.n))

            # Now we improve our policy again
            policy_output = pi_improvement(agent,
                                           pomdp,
                                           iterations=pi_per_step,
                                           log_every=log_every)
            info['policy_improvement_outputs'].append(policy_output)

            policy = agent.policy
            if agent.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
                agent.mem_params, unflat_policy = deconstruct_aug_policy(softmax(agent.pi_aug_params, axis=-1))

                O, M, A = unflat_policy.shape
                policy = unflat_policy.reshape(O * M, A)
                agent.pi_params = reverse_softmax(policy)

                # Plotting for memory iteration
                print(f"Learnt memory for iteration {mem_it}: \n"
                      f"{agent.memory}")

            # Plotting for policy improvement
            print(f"Learnt policy for iteration {mem_it}: \n"
                  f"{policy}")

    final_pomdp = pomdp

    # if we maximize lambda discrepancy, we need to do a final policy improvement step, with policy iteration.
    if agent.policy_optim_alg in ['discrep_max', 'discrep_min']:
        info[f'final_{agent.policy_optim_alg}_pi_params'] = agent.pi_params.copy()
        agent.reset_pi_params()

        og_policy_optim_algo = agent.policy_optim_alg

        # Change modes, run policy iteration
        agent.policy_optim_alg = 'policy_iter'
        print("Final policy improvement, after λ-discrep. optimization.")
        pi_improvement(agent, final_pomdp, iterations=pi_per_step, log_every=log_every)

        # Plotting for final policy iteration
        print(f"Learnt policy for final policy iteration: \n"
              f"{agent.policy}")

        # change our mode back
        agent.policy_optim_alg = og_policy_optim_algo

    if agent.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
        final_pomdp = memory_cross_product(agent.mem_params, init_pomdp)

    # here we calculate our value function for our final policy and final memory
    td_v_vals, td_q_vals = td_pe(agent.policy, final_pomdp)
    info['final_outputs'] = {'v': td_v_vals, 'q': td_q_vals}

    return info, agent

def pi_improvement(agent: AnalyticalAgent,
                   pomdp: POMDP,
                   iterations: int = 10000,
                   log_every: int = 1000,
                   progress_bar: bool = True) -> dict:
    """
    Policy improvement over multiple steps.
    :param agent:               Agent instance to run memory iteration.
    :param pomdp:                Initial abstract MDP for us to train our policy on.
    :param lr:                  Policy improvement learning rate
    :param iterations:          Number of policy improvement steps.
    :param log_every:           How many steps per log?
    """
    output = {}
    to_iterate = range(iterations)
    if progress_bar:
        to_iterate = trange(iterations)
    for it in to_iterate:
        output = agent.policy_improvement(pomdp)
        if it % log_every == 0:
            if 'v_0' in output:
                print(f"initial state value for iteration {it}: {output['v_0'].item():.4f}")
            if agent.policy_optim_alg in ['discrep_min', 'discrep_max']:
                print(f"discrep for pi iteration {it}: {output['loss'].item():.4f}")
            # elif agent.policy_optim_alg == 'policy_iter':
            #     outputs.append(output)
    return output

def mem_improvement(agent: AnalyticalAgent,
                    pomdp: POMDP,
                    iterations: int = 10000,
                    log_every: int = 1000,
                    progress_bar: bool = True) -> np.ndarray:
    """
    Memory improvement over multiple steps
    """
    memory_losses = []
    to_iterate = range(iterations)
    if progress_bar:
        to_iterate = trange(iterations)
    for it in to_iterate:
        loss = agent.memory_improvement(pomdp)
        if it % log_every == 0:
            print(f"Memory improvement loss for step {it}: {loss.item():.4f}")
            memory_losses.append(loss.item())
    return np.array(memory_losses)
