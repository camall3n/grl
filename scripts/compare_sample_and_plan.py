from definitions import ROOT_DIR
import pickle

import numpy as np
from jax.config import config
from pathlib import Path

from grl.agents.actorcritic import ActorCritic
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.memory import memory_cross_product
from scripts.learning_agent.memory_iteration import converge_value_functions, parse_args

np.set_printoptions(precision=3, suppress=True)
config.update('jax_platform_name', 'cpu')

args = parse_args()
args.min_mem_opt_replay_size = args.replay_buffer_size
del args.f

spec_name = "tmaze_eps_hyperparams"
# spec_name = "simple_chain"
corridor_length = 5
discount = 0.9
junction_up_pi = 2 / 3
epsilon = 0.2
error_type = 'mse'

print(f"Running sample-based and analytical comparison for Q-values on {spec_name}")
spec = load_spec(spec_name,
                 memory_id=str(16),
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
    lambda_0=args.lambda0,
    lambda_1=args.lambda1,
    learning_rate=args.learning_rate,
    n_mem_entries=0,
    replay_buffer_size=args.replay_buffer_size,
    mellowmax_beta=10.,
    discrep_loss=error_type,
    study_name='compare_sample_and_plan_04/' + args.study_name,
)

agent.set_policy(pi, logits=False)

agent.add_memory()
# agent.reset_memory()
# mem_params = agent.memory_logits
mem_params = spec['mem_params']
agent.set_memory(mem_params, logits=True)
mem_aug_mdp = memory_cross_product(mem_params, env)

# analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(pi.repeat(2, axis=0), mem_aug_mdp)
# analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(pi, env)
lstd_v0, lstd_q0, _ = lstdq_lambda(pi.repeat(2, axis=0), mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1, _ = lstdq_lambda(pi.repeat(2, axis=0), mem_aug_mdp, lambda_=args.lambda1)

converge_value_functions(agent, env, args.n_samples_per_policy)

samp_q0 = agent.q_td.q
samp_q1 = agent.q_mc.q

print(f"Lambda = {args.lambda0}")
print("Sample-based:")
print(samp_q0)

print("Analytical:")
print(lstd_q0)

print('---------------------------')
print(f"Lambda = {args.lambda1}")
print("Sample-based:")
print(samp_q1)

print("Analytical:")
print(lstd_q1)

print("done")

info = {
    'lstd_q0': lstd_q0,
    'lstd_q1': lstd_q1,
    'samp_q0': samp_q0,
    'samp_q1': samp_q1,
    'args': args.__dict__
}

# #%%
#
# print("Lambda = 0")
# print(np.abs(samp_q0 - lstd_q0).mean())
#
# print('---------------------------')
# print("Lambda = 0.8")
# print(np.abs(samp_q8 - lstd_q8).mean())
#
# print('---------------------------')
# print("Lambda = 0.9")
# print(np.abs(samp_q9 - lstd_q9).mean())
#
# print('---------------------------')
# print("Lambda = 0.99999")
# print(np.abs(samp_q1 - lstd_q1).mean())
#
# print("done")

with open(agent.study_dir + '/sample_vs_plan.pkl', 'wb') as file:
    pickle.dump(info, file)
# buffer_dir = Path(ROOT_DIR, 'scripts', 'results', 'sample_based')
#
# fname = f'replaymemory_corridor({corridor_length})_eps({epsilon})_size({buffer_size})'
# ext = '.pkl'
# print(f"Saving to {buffer_dir / (fname + ext)}")
# agent.replay.save(buffer_dir, filename=fname, extension=ext)
