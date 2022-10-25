import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import softmax
from tqdm import trange

from grl.analytical_agent import AnalyticalAgent
from grl.policy_eval import PolicyEval
from grl.mdp import AbstractMDP, MDP
from grl.memory import memory_cross_product
from grl.utils import golrot_init
from grl.vi import td_pe

def lambda_discrep_measures(amdp: AbstractMDP, pi: jnp.ndarray):
    amdp_pe = PolicyEval(amdp)
    state_vals, mc_vals, td_vals = amdp_pe.run(pi)
    pi_occupancy = amdp_pe.get_occupancy(pi)
    pr_oa = (pi_occupancy @ amdp.phi * pi.T)
    discrep = {
        'v': (mc_vals['v'] - td_vals['v'])**2,
        'q': (mc_vals['q'] - td_vals['q'])**2,
    }
    discrep['q_sum'] = (discrep['q'] * pr_oa).sum()
    return discrep

def run_memory_iteration(spec: dict, pi_lr: float = 1., mi_lr: float = 1.,
                         policy_optim_alg: str = 'pi', mi_iterations: int = 1,
                         rand_key: jax.random.PRNGKey = None):
    """
    Runs interspersing memory iteration and policy improvement.
    """
    assert 'mem_params' in spec.keys() and spec['mem_params'] is not None
    mem_params = spec['mem_params']

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    # initialize policy params
    pi_params = golrot_init(spec['Pi_phi'][0].shape)
    initial_policy = softmax(pi_params, axis=-1)

    agent = AnalyticalAgent(pi_params, mem_params=mem_params, rand_key=rand_key,
                            policy_optim_alg=policy_optim_alg)

    info, agent = memory_iteration(agent, amdp, pi_lr=pi_lr, mi_lr=mi_lr, mi_iterations=mi_iterations)

    # we get lambda discrepancies here

    # initial policy lambda-discrepancy
    info['initial_discrep'] = lambda_discrep_measures(amdp, initial_policy)
    info['initial_improvement_discrep'] = lambda_discrep_measures(amdp, info['initial_improvement_policy'])

    # Initial memory amdp w/ initial improvement policy discrep
    if 'initial_mem_params' in info and info['initial_mem_params'] is not None:
        init_mem_amdp = memory_cross_product(amdp, info['initial_mem_params'])
        info['initial_mem_discrep'] = lambda_discrep_measures(init_mem_amdp, info['initial_expanded_improvement_policy'])

    # Final memory w/ final policy discrep
    final_mem_amdp = memory_cross_product(amdp, agent.mem_params)
    info['final_mem_discrep'] = lambda_discrep_measures(final_mem_amdp, agent.policy)

    return info, agent

def memory_iteration(agent: AnalyticalAgent, init_amdp: AbstractMDP,
                         pi_lr: float = 1., mi_lr: float = 0.1,
                         pi_per_step: int = 50000, mi_per_step: int = 50000,
                         mi_iterations: int = 1,
                         log_every: int = 1000):
    info = {'policy_improvement_outputs': [], 'mem_loss': []}

    # initial policy improvement
    print("Initial policy improvement step")
    initial_outputs = pi_improvement(agent, init_amdp, lr=pi_lr, iterations=pi_per_step, log_every=log_every)
    info['policy_improvement_outputs'].append(initial_outputs)
    info['initial_improvement_policy'] = agent.policy.copy()
    info['initial_mem_params'] = agent.mem_params
    print(f"Starting (unexpanded) policy: \n{agent.policy}")

    # we have to set our policy over our memory MDP now
    # we do so with a random policy given new memory bit
    agent.new_pi_over_mem()
    info['initial_expanded_improvement_policy'] = agent.policy.copy()
    print(f"Starting (expanded) policy: \n{agent.policy}")
    print(f"Starting memory: \n{agent.memory}")

    amdp_mem = None

    for mem_it in range(mi_iterations):
        print(f"Start MI iteration {mem_it}")
        mem_loss = mem_improvement(agent, init_amdp, lr=mi_lr, iterations=mi_per_step, log_every=log_every)
        info['mem_loss'].append(mem_loss)

        # Plotting for memory iteration
        print(f"Learnt memory for iteration {mem_it}: \n"
              f"{softmax(agent.mem_params, axis=-1)}")

        # Make a NEW memory AMDP
        amdp_mem = memory_cross_product(init_amdp, agent.mem_params)

        # Now we need to reset policy params, with the appropriate new shape
        if agent.pi_params.shape != (amdp_mem.n_obs, amdp_mem.n_actions):
            agent.reset_pi_params((amdp_mem.n_obs, amdp_mem.n_actions))

        # Now we improve our policy again
        policy_output = pi_improvement(agent, amdp_mem, lr=pi_lr, iterations=pi_per_step, log_every=log_every)
        info['policy_improvement_outputs'].append(policy_output)

        # Plotting for policy improvement
        print(f"Learnt policy for iteration {mem_it}: \n"
              f"{agent.policy}")

    final_amdp = init_amdp if amdp_mem is None else amdp_mem

    # here we calculate our value function for our final policy and final memory
    eval_policy = info['initial_improvement_policy'] if amdp_mem is None else agent.policy
    td_v_vals, td_q_vals = td_pe(eval_policy, final_amdp.T, final_amdp.R, final_amdp.phi, final_amdp.p0, final_amdp.gamma)
    info['final_outputs'] = {'v': td_v_vals, 'q': td_q_vals}

    return info, agent

def pi_improvement(agent: AnalyticalAgent, amdp: AbstractMDP,
           lr: float = 1., iterations: int = 10000,
           log_every: int = 1000) -> dict:
    """
    Policy improvement over multiple steps
    """
    output = {}
    for it in trange(iterations):
        output = agent.policy_improvement(amdp, lr)
        if it % log_every == 0:
            if agent.policy_optim_alg == 'pg':
                print(f"initial state value for iteration {it}: {output['v_0'].item():.4f}")
            if agent.policy_optim_alg == 'dm':
                print(f"discrep for pi iteration {it}: {output['loss'].item():.4f}")
            # elif agent.policy_optim_alg == 'pi':
            #     outputs.append(output)
    return output

def mem_improvement(agent: AnalyticalAgent, amdp: AbstractMDP,
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
