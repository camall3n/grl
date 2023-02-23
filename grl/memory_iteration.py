import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import softmax
from tqdm import trange
from functools import partial

from grl.agents.analytical import AnalyticalAgent
from grl.utils.lambda_discrep import lambda_discrep_measures
from grl.mdp import AbstractMDP, MDP
from grl.memory import memory_cross_product
from grl.utils.math import glorot_init, greedify
from grl.utils.loss import discrep_loss
from grl.vi import td_pe

def run_memory_iteration(spec: dict,
                         pi_lr: float = 1.,
                         mi_lr: float = 1.,
                         policy_optim_alg: str = 'pi',
                         mi_iterations: int = 1,
                         mi_steps: int = 50000,
                         pi_steps: int = 50000,
                         rand_key: jax.random.PRNGKey = None,
                         error_type: str = 'l2',
                         value_type: str = 'q',
                         objective: str = 'discrep',
                         alpha: float = 1.,
                         pi_params: jnp.ndarray = None,
                         epsilon: float = 0.1,
                         flip_count_prob: bool = False
):
    """
    Wrapper function for the Memory Iteration algorithm.
    Memory iteration intersperses memory improvement and policy improvement.
    :param spec:                POMDP specification.
    :param pi_lr:               Policy improvement learning rate
    :param mi_lr:               Memory improvement learning rate
    :param policy_optim_alg:    Which policy improvement algorithm do we use?
        (dm: discrepancy maximization | pi: policy iteration | pg: policy gradient)
    :param mi_iterations:       How many memory iterations do we do?
    :param mi_steps:            Number of memory improvement steps PER memory iteration step.
    :param pi_steps:            Number of policy improvement steps PER memory iteration step.
    :param flip_count_prob:     Do we flip our count probabilities over observations for memory loss?.
    """
    assert 'mem_params' in spec.keys() and spec['mem_params'] is not None
    mem_params = spec['mem_params']

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    assert amdp.current_state is None, \
        f"AbstractMDP should be stateless and current_state should be None, got {amdp.current_state} instead"

    # initialize policy params
    if 'Pi_phi' not in spec or spec['Pi_phi'] is None:
        pi_phi_shape = (spec['phi'].shape[-1], spec['T'].shape[0])
    else:
        pi_phi_shape = spec['Pi_phi'][0].shape

    # If pi_params is initialized, we start with some given
    # policy params and don't do the first policy improvement step.
    init_pi_improvement = False
    if pi_params is None:
        init_pi_improvement = True
        pi_params = glorot_init(pi_phi_shape, scale=0.2)
    initial_policy = softmax(pi_params, axis=-1)

    agent = AnalyticalAgent(pi_params,
                            rand_key,
                            mem_params=mem_params,
                            policy_optim_alg=policy_optim_alg,
                            error_type=error_type,
                            value_type=value_type,
                            objective=objective,
                            alpha=alpha,
                            epsilon=epsilon,
                            flip_count_prob=flip_count_prob)

    info, agent = memory_iteration(agent,
                                   amdp,
                                   pi_lr=pi_lr,
                                   mi_lr=mi_lr,
                                   mi_iterations=mi_iterations,
                                   pi_per_step=pi_steps,
                                   mi_per_step=mi_steps,
                                   init_pi_improvement=init_pi_improvement)

    discrep_loss_fn = partial(discrep_loss, value_type=value_type, error_type=error_type, alpha=alpha)
    get_measures = partial(lambda_discrep_measures, discrep_loss_fn=discrep_loss_fn)

    info['initial_policy'] = initial_policy
    # we get lambda discrepancies here
    # initial policy lambda-discrepancy
    info['initial_policy_stats'] = get_measures(amdp, initial_policy)
    info['initial_improvement_stats'] = get_measures(amdp, info['initial_improvement_policy'])
    greedy_initial_improvement_policy = greedify(info['initial_improvement_policy'])
    info['greedy_initial_improvement_stats'] = get_measures(amdp, greedy_initial_improvement_policy)


    if policy_optim_alg == 'dm' or not init_pi_improvement:
        info['td_optimal_policy_stats'] = get_measures(amdp, info['td_optimal_memoryless_policy'])
        info['greedy_td_optimal_policy_stats'] = get_measures(amdp, greedify(info['td_optimal_memoryless_policy']))

    # Initial memory amdp w/ initial improvement policy discrep
    if 'initial_mem_params' in info and info['initial_mem_params'] is not None:
        init_mem_amdp = memory_cross_product(info['initial_mem_params'], amdp)
        info['initial_mem_stats'] = get_measures(
            init_mem_amdp, info['initial_expanded_improvement_policy'])

    # Final memory w/ final policy discrep
    final_mem_amdp = memory_cross_product(agent.mem_params, amdp)
    info['final_mem_stats'] = get_measures(final_mem_amdp, agent.policy)
    greedy_final_policy = greedify(agent.policy)
    info['greedy_final_mem_stats'] = get_measures(final_mem_amdp, greedy_final_policy)

    def perf_from_stats(stats: dict) -> float:
        return np.dot(stats['state_vals_v'], stats['p0']).item()

    print("Finished Memory Iteration.")
    print(f"Initial policy performance: {perf_from_stats(info['initial_policy_stats']):.4f}")
    print(f"Initial improvement performance: {perf_from_stats(info['initial_improvement_stats']):.4f}")
    if 'greedy_td_optimal_policy_stats' in info:
        print(f"TD-optimal policy performance: {perf_from_stats(info['greedy_td_optimal_policy_stats']):.4f}")
    print(f"Final performance after MI: {perf_from_stats(info['greedy_final_mem_stats']):.4f}")

    return info, agent

def memory_iteration(
    agent: AnalyticalAgent,
    init_amdp: AbstractMDP,
    pi_lr: float = 1.,
    mi_lr: float = 1,
    pi_per_step: int = 50000,
    mi_per_step: int = 50000,
    mi_iterations: int = 1,
    init_pi_improvement: bool = True,
    log_every: int = 1000,
):
    """
    The memory iteration algorithm. This algorithm flips between improving
    policy parameters to maximize return, and memory parameters to minimize lambda discrepancy.
    Each step of memory iteration includes multiple steps of memory improvement and policy improvement.
    :param agent:               Agent instance to run memory iteration.
    :param init_amdp:           Initial abstract MDP for us to train our agent on.
    :param pi_lr:               Policy improvement learning rate
    :param mi_lr:               Memory improvement learning rate
    :param pi_per_step:         Number of policy improvement steps PER memory iteration step.
    :param mi_per_step:         Number of memory improvement steps PER memory iteration step.
    :param mi_iterations:       Number of steps of memory iteration to perform.
    :param init_pi_improvement: Do we start out with an initial policy improvement step?
    :param log_every:           How often do we log stats?
    """
    info = {'policy_improvement_outputs': [], 'mem_loss': []}
    td_v_vals, td_q_vals = td_pe(agent.policy, init_amdp)
    info['initial_values'] = {'v': td_v_vals, 'q': td_q_vals}

    if agent.policy_optim_alg in ['discrep_max', 'discrep_min'] and not init_pi_improvement:
        initial_pi_params = agent.pi_params.copy()

        og_policy_optim_algo = agent.policy_optim_alg

        # Change modes, run policy iteration
        agent.policy_optim_alg = 'policy_iter'
        print(f"Calculating TD-optimal memoryless policy over {pi_per_step} steps")
        pi_improvement(agent, init_amdp, lr=pi_lr, iterations=pi_per_step, log_every=log_every)
        info['td_optimal_memoryless_policy'] = agent.policy.copy()
        print(f"Converged to TD-optimal memoryless policy: \n{agent.policy}\n")

        # reset our policy again
        agent.pi_params = initial_pi_params
        agent.policy_optim_alg = og_policy_optim_algo

    if init_pi_improvement:
        # initial policy improvement
        print("Initial policy improvement step")
        initial_outputs = pi_improvement(agent,
                                         init_amdp,
                                         lr=pi_lr,
                                         iterations=pi_per_step,
                                         log_every=log_every)
        info['policy_improvement_outputs'].append(initial_outputs)

    info['initial_improvement_policy'] = agent.policy.copy()
    info['initial_mem_params'] = agent.mem_params
    print(f"Starting (unexpanded) policy: \n{agent.policy}\n")

    # we have to set our policy over our memory MDP now
    # we do so with a random/copied policy given new memory bit
    agent.new_pi_over_mem()
    info['initial_expanded_improvement_policy'] = agent.policy.copy()
    print(f"Starting (expanded) policy: \n{agent.policy}\n")
    print(f"Starting memory: \n{agent.memory}\n")

    amdp_mem = None

    for mem_it in range(mi_iterations):
        print(f"Start MI {mem_it}")
        mem_loss = mem_improvement(agent,
                                   init_amdp,
                                   lr=mi_lr,
                                   iterations=mi_per_step,
                                   log_every=log_every)
        info['mem_loss'].append(mem_loss)

        # Plotting for memory iteration
        print(f"Learnt memory for iteration {mem_it}: \n"
              f"{agent.memory}")

        # Make a NEW memory AMDP
        amdp_mem = memory_cross_product(agent.mem_params, init_amdp)

        if pi_per_step > 0:
            # reset our policy parameters
            agent.reset_pi_params((amdp_mem.n_obs, amdp_mem.n_actions))

            # Now we improve our policy again
            policy_output = pi_improvement(agent,
                                           amdp_mem,
                                           lr=pi_lr,
                                           iterations=pi_per_step,
                                           log_every=log_every)
            info['policy_improvement_outputs'].append(policy_output)

            # Plotting for policy improvement
            print(f"Learnt policy for iteration {mem_it}: \n"
                  f"{agent.policy}")

    final_amdp = init_amdp if amdp_mem is None else amdp_mem

    # if we maximize lambda discrepancy, we need to do a final policy improvement step, with policy iteration.
    if agent.policy_optim_alg == 'dm':
        info['final_dm_pi_params'] = agent.pi_params.copy()
        agent.reset_pi_params()

        og_policy_optim_algo = agent.policy_optim_alg

        # Change modes, run policy iteration
        agent.policy_optim_alg = 'pi'
        print("Final policy improvement, after λ-discrep. max.")
        pi_improvement(agent, final_amdp, lr=pi_lr, iterations=pi_per_step, log_every=log_every)

        # Plotting for final policy iteration
        print(f"Learnt policy for final policy iteration: \n"
              f"{agent.policy}")

        # change our mode back
        agent.policy_optim_alg = og_policy_optim_algo

    # here we calculate our value function for our final policy and final memory
    eval_policy = info['initial_improvement_policy'] if amdp_mem is None else agent.policy
    td_v_vals, td_q_vals = td_pe(eval_policy, final_amdp)
    info['final_outputs'] = {'v': td_v_vals, 'q': td_q_vals}

    return info, agent

def pi_improvement(agent: AnalyticalAgent,
                   amdp: AbstractMDP,
                   lr: float = 1.,
                   iterations: int = 10000,
                   log_every: int = 1000,
                   progress_bar: bool = True) -> dict:
    """
    Policy improvement over multiple steps.
    :param agent:               Agent instance to run memory iteration.
    :param amdp:                Initial abstract MDP for us to train our policy on.
    :param lr:                  Policy improvement learning rate
    :param iterations:          Number of policy improvement steps.
    :param log_every:           How many steps per log?
    """
    output = {}
    to_iterate = range(iterations)
    if progress_bar:
        to_iterate = trange(iterations)
    for it in to_iterate:
        output = agent.policy_improvement(amdp, lr)
        if it % log_every == 0:
            if agent.policy_optim_alg == 'pg':
                print(f"initial state value for iteration {it}: {output['v_0'].item():.4f}")
            if agent.policy_optim_alg == 'dm':
                print(f"discrep for pi iteration {it}: {output['loss'].item():.4f}")
            # elif agent.policy_optim_alg == 'pi':
            #     outputs.append(output)
    return output

def mem_improvement(agent: AnalyticalAgent,
                    amdp: AbstractMDP,
                    lr: float = 1.,
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
        loss = agent.memory_improvement(amdp, lr)
        if it % log_every == 0:
            print(f"Memory improvement loss for step {it}: {loss.item():.4f}")
            memory_losses.append(loss.item())
    return np.array(memory_losses)
