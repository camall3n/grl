import numpy as np
from jax.nn import softmax
from tqdm import trange

from grl.analytical_agent import AnalyticalAgent
from grl.mdp import AbstractMDP
from grl.memory import memory_cross_product

def run_memory_iteration(agent: AnalyticalAgent, init_amdp: AbstractMDP,
                         pi_lr: float = 0.5, mi_lr: float = 0.1,
                         pi_per_step: int = 10000, mi_per_step: int = 25000,
                         mpi_iterations: int = 2,
                         log_every: int = 1000):
    log = {'policy_improvement_outputs': [], 'mem_loss': []}

    # initial policy improvement
    print("Initial policy improvement step")
    initial_outputs = run_pi(agent, init_amdp, lr=pi_lr, iterations=pi_per_step, log_every=log_every)
    log['policy_improvement_outputs'].append(initial_outputs)

    # we have to set our policy over our memory MDP now
    # we do so with a random policy given new memory bit
    agent.new_pi_over_mem()
    print(f"Starting policy: \n{agent.policy}")
    print(f"Starting memory: \n{agent.memory}")

    for mem_it in range(1, mpi_iterations + 1):
        print(f"Start MPI iteration {mem_it}")
        mem_loss = run_mi(agent, init_amdp, lr=mi_lr, iterations=mi_per_step, log_every=log_every)
        log['mem_loss'].append(mem_loss)

        # Plotting for memory iteration
        print(f"Learnt memory for iteration {mem_it}: \n"
              f"{softmax(agent.mem_params, axis=-1)}")

        # Make a NEW memory AMDP
        amdp_mem = memory_cross_product(init_amdp, agent.mem_params)

        # Now we need to reset policy params, with the appropriate new shape
        if agent.pi_params.shape != (amdp_mem.n_obs, amdp_mem.n_actions):
            agent.reset_pi_params((amdp_mem.n_obs, amdp_mem.n_actions))

        # Now we improve our policy again
        policy_output = run_pi(agent, amdp_mem, lr=pi_lr, iterations=pi_per_step * 2, log_every=log_every)
        log['policy_improvement_outputs'].append(policy_output)

        # Plotting for policy improvement
        print(f"Learnt policy for iteration {mem_it}: \n"
              f"{agent.policy}")

    return log, agent

def run_pi(agent: AnalyticalAgent, amdp: AbstractMDP,
           lr: float = 1., iterations: int = 10000,
           log_every: int = 1000) -> np.ndarray:
    """
    Policy improvement over multiple steps
    """
    outputs = []
    for it in trange(iterations):
        output = agent.policy_improvement(amdp, lr)
        if it % log_every == 0:
            if agent.policy_optim_alg == 'pg':
                print(f"initial state value for iteration {it}: {output.item():.4f}")
                outputs.append(output.item())
            elif agent.policy_optim_alg == 'pi':
                outputs.append(output)
    print(f"learnt policy: {agent.policy}")
    return np.array(outputs)

def run_mi(agent: AnalyticalAgent, amdp: AbstractMDP,
           lr: float = 1., iterations: int = 10000,
           log_every: int = 1000) -> np.ndarray:
    """
    Memory improvement over multiple steps
    """
    memory_losses = []
    for it in trange(iterations):
        loss = agent.memory_improvement(amdp, lr)
        if it % log_every == 0:
            print(f"Memory improvement loss for step {it}: {loss.item():.4f}")
            memory_losses.append(loss.item())
    return np.array(memory_losses)
