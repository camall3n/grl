from jax.config import config

config.update('jax_platform_name', 'cpu')

from grl.agents.actorcritic import ActorCritic
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import analytical_pe
from grl.memory import memory_cross_product

from learning_agent.memory_iteration import converge_value_functions

if __name__ == "__main__":
    spec_name = "tmaze_eps_hyperparams"
    # spec_name = "simple_chain"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 2/3
    epsilon = 0.0
    error_type = 'l2'

    # Sampling agent hyperparams
    buffer_size = int(4e6)

    print(f"Running sample-based and analytical comparison for Q-values on {spec_name}")
    spec = load_spec(spec_name,
                     memory_id=None,
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]


    agent = ActorCritic(
        n_obs=env.n_obs,
        n_actions=env.n_actions,
        gamma=env.gamma,
        n_mem_entries=0,
        replay_buffer_size=buffer_size,
        mellowmax_beta=10.,
        discrep_loss=error_type,
    )

    agent.set_policy(pi, logits=False)

    # agent.add_memory()
    # agent.reset_memory()
    # mem_params = agent.memory_logits
    # mem_aug_mdp = memory_cross_product(mem_params, env)

    # analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(pi.repeat(2, axis=0), mem_aug_mdp)
    analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(pi, env)

    converge_value_functions(agent, env, update_policy=False)

    print("Sample-based Q-TD values:")
    print(agent.q_td.q)

    print("Analytical Q-TD values:")
    print(analytical_td_vals['q'])

    print("Sample-based Q-MC values:")
    print(agent.q_mc.q)

    print("Analytical Q-MC values:")
    print(analytical_mc_vals['q'])

    print("done")
