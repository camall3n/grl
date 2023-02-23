from grl.agents.actorcritic import ActorCritic
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import analytical_pe

from learning_agent.memory_iteration import converge_value_functions

if __name__ == "__main__":
    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 2/3
    epsilon = 0.1
    error_type = 'l2'

    # Sampling agent hyperparams
    buffer_size = int(4e5)

    spec = load_spec(spec_name,
                     memory_id=None,
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]

    analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(pi, env)

    agent = ActorCritic(
        n_obs=env.n_obs,
        n_actions=env.n_actions,
        gamma=env.gamma,
        n_mem_entries=0,
        replay_buffer_size=buffer_size,
        mellowmax_beta=10.,
        discrep_loss=error_type,
    )

    agent.set_policy(pi)
    agent.add_memory()
    agent.reset_memory()
    converge_value_functions(agent, env, update_policy=False)

    print("Sample-based Q-TD values:")
    print(agent.q_td)

    print("Analytical Q-TD values:")
    print(analytical_td_vals['q'])

    print("Sample-based Q-MC values:")
    print(agent.q_mc)

    print("Analytical Q-MC values:")
    print(analytical_mc_vals['q'])

