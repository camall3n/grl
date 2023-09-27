import copy
import json
import os
from pprint import pprint
import sys

import jax
from jax.config import config
import numpy as np
from tqdm import tqdm, trange

jax.config.update("jax_enable_x64", True)

from grl.agent.actorcritic import ActorCritic
from grl.agent.analytical import AnalyticalAgent
from grl.memory_iteration import pi_improvement
from grl.environment import load_spec
from grl.mdp import MDP, POMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.loss import discrep_loss
from grl.utils.discrete_search import generate_hold_mem_fn
from grl.memory import memory_cross_product
from scripts.learning_agent.memory_iteration import parse_args

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
config.update('jax_platform_name', 'cpu')

args = parse_args()
np.random.seed(args.seed)

args.min_mem_opt_replay_size = args.replay_buffer_size
del args.f

if args.mem_optimizer == 'annealing' and args.annealing_tmin > args.annealing_tmax:
    print('Annealing tmin > tmax. Skipping run.')
    sys.exit()

# args.env = 'cheese.95'
# args.env = 'slippery-tmaze'
# args.env = '4x3.95'
# args.env = 'shuttle.95'
# args.env = 'example_7'
#
n_pi_iterations = 5000

reward_range_dict = {
    'cheese.95': (10.0, 0),
    'tiger-alt-start': (10.0, -100.0),
    'network': (80.0, -40.0),
    'slippery-tmaze': (4.0, -0.1),
    'tmaze_5_two_thirds_up': (4.0, -0.1),
    'example_7': (1.0, 0.0),
    '4x3.95': (1.0, -1.0),
    'shuttle.95': (10.0, -3.0),
    'paint.95': (1.0, -1.0),
    'bridge-repair': (4018, 0),
    'hallway': (1.0, 0),
}
reward_scale = args.reward_scale
if args.normalize_reward_range and args.env in reward_range_dict:
    reward_scale = 1 / (reward_range_dict[args.env][0] - reward_range_dict[args.env][1])

# Env stuff
spec = load_spec(args.env, memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
mdp.R *= reward_scale
env = POMDP(mdp, spec['phi'])

learning_agent = ActorCritic(
    n_obs=env.observation_space.n,
    n_actions=env.action_space.n,
    gamma=env.gamma,
    lambda_0=args.lambda0,
    lambda_1=args.lambda1,
    learning_rate=args.learning_rate,
    n_mem_entries=0,
    n_mem_values=args.n_memory_states,
    replay_buffer_size=args.replay_buffer_size,
    mem_optimizer=args.mem_optimizer,
    ignore_queue_priority=(not args.enable_priority_queue),
    annealing_should_sample_hyperparams=args.annealing_should_sample_hyperparams,
    annealing_tmax=args.annealing_tmax,
    annealing_tmin=args.annealing_tmin,
    annealing_progress_fraction_at_tmin=args.annealing_progress_fraction_at_tmin,
    n_annealing_repeats=args.n_annealing_repeats,
    prune_if_parent_suboptimal=False,
    mellowmax_beta=10.,
    discrep_loss='mse',
    study_name='compare_sample_and_plan_04/' + args.study_name,
    override_mem_eval_with_analytical_env=env,
    analytical_lambda_discrep_noise=0.00,
)

planning_agent = AnalyticalAgent(
    pi_params=learning_agent.policy_logits,
    rand_key=jax.random.PRNGKey(args.seed),
    pi_lr=args.policy_optim_lr,
    mem_params=learning_agent.memory_logits,
    value_type='q',
    policy_optim_alg=(None if args.init_policy_randomly else args.policy_optim_alg),
)

# Policy stuff
if not args.init_policy_randomly:
    pi_improvement(planning_agent, env, iterations=n_pi_iterations)
    learning_agent.set_policy(planning_agent.pi_params, logits=True)
else:
    largest_discrep = 0
    largest_discrep_policy = None
    for i in trange(args.n_random_policies):
        learning_agent.reset_policy()
        policy = learning_agent.policy_probs
        policy_lamdba_discrep = discrep_loss(policy, env)[0].item()
        if policy_lamdba_discrep > largest_discrep:
            largest_discrep = policy_lamdba_discrep
            largest_discrep_policy = policy
    learning_agent.set_policy(policy, logits=False)
learning_agent.add_memory()
pi_aug = learning_agent.policy_probs

# Memory stuff
mem_probs = generate_hold_mem_fn(learning_agent.n_actions, learning_agent.n_obs,
                                 learning_agent.n_mem_states)
learning_agent.set_memory(mem_probs, logits=False)
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)

# Value stuff
def get_start_obs_value(pi, mdp):
    mdp = copy.deepcopy(mdp)
    mdp.R /= reward_scale
    value_fn, _, _ = lstdq_lambda(pi, mdp, lambda_=args.lambda1)
    return (value_fn @ (mdp.p0 @ mdp.phi)).item()

start_value = get_start_obs_value(pi_aug, mem_aug_mdp)
initial_discrep = discrep_loss(pi_aug, mem_aug_mdp)[0].item()

# Search stuff
info = learning_agent.optimize_memory(args.n_memory_trials)
if learning_agent.mem_optimizer == "queue":
    M = learning_agent.n_mem_states
    O = learning_agent.n_obs
    A = learning_agent.n_actions
    n_mem_fns = M**(M * O * A)
    print(f'Total memory functions: {n_mem_fns}')
    print(f'Number evaluated: {info["n_evals"]}')

# plt.plot(range(len(info['discreps'])), info['discreps'])
# plt.show()

# Search results stuff
print()
# print(f'Best node: \n{learning_agent.memory_probs}\n')
print(f'Best discrep: {info["best_discrep"]}')

# Final performance stuff
learning_agent.reset_policy()
planning_agent = AnalyticalAgent(
    pi_params=learning_agent.policy_logits,
    rand_key=jax.random.PRNGKey(args.seed + 10000),
    pi_lr=args.policy_optim_lr,
    mem_params=learning_agent.memory_logits,
    value_type='q',
    policy_optim_alg=args.policy_optim_alg,
)
planning_agent.reset_pi_params((mem_aug_mdp.observation_space.n, mem_aug_mdp.action_space.n))
pi_improvement(planning_agent, mem_aug_mdp, iterations=n_pi_iterations)
learning_agent.set_policy(planning_agent.pi_params, logits=True)

end_value = get_start_obs_value(learning_agent.policy_probs, mem_aug_mdp)
print(f'Start value: {start_value}')
print(f'End value: {end_value}')

# Save stuff
results = {
    'env': args.env,
    'study_name': args.study_name,
    'trial_id': args.trial_id,
    'seed': args.seed,
    'n_mem_states': args.n_memory_states,
    'policy_optim_alg': args.policy_optim_alg,
    'init_policy_randomly': args.init_policy_randomly,
    'n_random_policies': args.n_random_policies,
    'mem_optimizer': args.mem_optimizer,
    'enable_priority_queue': args.enable_priority_queue,
    'tmax': args.annealing_tmax,
    'tmin': args.annealing_tmin,
    'progress_fraction_at_tmin': args.annealing_progress_fraction_at_tmin,
    'n_repeats': args.n_annealing_repeats,
    'n_iters': args.n_memory_trials,
    'start_value': start_value,
    'end_value': end_value,
    'initial_discrep': initial_discrep,
    'best_discrep': info['best_discrep'],
}
pprint(results, sort_dicts=False)
results['optimizer_info'] = info

dirname = f'results/discrete/{args.study_name}/{args.env}/{args.trial_id}'
json_filename = dirname + '/discrete_oracle.json'
npy_filename = dirname + '/memory.npy'
os.makedirs(dirname, exist_ok=True)
with open(json_filename, 'w') as f:
    json.dump(results, f)

np.save(npy_filename, learning_agent.memory_probs)

# #%%
# plt.plot(range(len(info['temps'])), [b for b in info['temps']])
# plt.show()
#
# #%%
# plt.plot(range(len(info['accept_probs'])), [b for b in info['accept_probs']])
# plt.show()
