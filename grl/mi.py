import numpy as np
from jax.nn import softmax
from tqdm import trange

from grl.analytical_agent import AnalyticalAgent
from grl.mdp import AbstractMDP
from grl.memory import memory_cross_product

def run_memory_iteration(agent: AnalyticalAgent, init_amdp: AbstractMDP,
                         pi_lr: float = 0.5, mi_lr: float = 0.1,
                         pi_per_step: int = 25000, mi_per_step: int = 25000,
                         mpi_iterations: int = 20,
                         log_every: int = 1000):
    log = {'v0': [], 'mem_loss': []}

    # initial policy improvement
    print("Initial policy improvement step")
    initial_v0s = run_pi(agent, init_amdp, lr=pi_lr, iterations=pi_per_step, log_every=log_every)
    log['v0'].append(initial_v0s)

    # we have to set our policy over our memory MDP now
    agent.repeat_pi_over_mem()
    print(f"Starting Policy: \n{softmax(agent.pi_params, axis=-1)}")

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
        v0 = run_pi(agent, amdp_mem, lr=pi_lr, iterations=pi_per_step * 2, log_every=log_every)
        log['v0'].append(v0)

        # Plotting for policy improvement
        print(f"Learnt policy for iteration {mem_it}: \n"
              f"{softmax(agent.pi_params, axis=-1)}")

    return log, agent

def run_pi(agent: AnalyticalAgent, amdp: AbstractMDP,
           lr: float = 1., iterations: int = 10000,
           log_every: int = 1000) -> np.ndarray:
    """
    Policy improvement over multiple steps
    """
    v_0s = []
    for it in trange(iterations):
        v_0 = agent.policy_improvement(amdp, lr)
        if it % log_every == 0:
            print(f"initial state value for iteration {it}: {v_0:.4f}")
            v_0s.append(v_0.item())
    return np.array(v_0s)

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
