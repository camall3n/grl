import json
import os

import numpy as np
from jax.config import config

from grl.agents.actorcritic import ActorCritic
from grl.agents.analytical import AnalyticalAgent
from grl.memory_iteration import pi_improvement
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.loss import discrep_loss
from grl.utils.discrete_search import generate_hold_mem_fn
from grl.memory import memory_cross_product
from scripts.learning_agent.memory_iteration import parse_args

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
config.update('jax_platform_name', 'cpu')

args = parse_args()
args.min_mem_opt_replay_size = args.replay_buffer_size
del args.f

# Env stuff
spec = load_spec(args.env, memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])

learning_agent = ActorCritic(
    n_obs=env.n_obs,
    n_actions=env.n_actions,
    gamma=env.gamma,
    lambda_0=args.lambda0,
    lambda_1=args.lambda1,
    learning_rate=args.learning_rate,
    n_mem_entries=0,
    replay_buffer_size=args.replay_buffer_size,
    mem_optimizer='annealing',
    mellowmax_beta=10.,
    discrep_loss='mse',
    study_name='compare_sample_and_plan_04/' + args.study_name,
    override_mem_eval_with_analytical_env=env,
)

planning_agent = AnalyticalAgent(
    pi_params=learning_agent.policy_probs,
    rand_key=None,
    mem_params=learning_agent.memory_probs,
    value_type='q',
)

# Policy stuff
pi_improvement(planning_agent, env, iterations=100, lr=0.1)
learning_agent.set_policy(planning_agent.pi_params, logits=True)
learning_agent.add_memory()
pi_aug = learning_agent.policy_probs

# Memory stuff
mem_probs = generate_hold_mem_fn(learning_agent.n_actions, learning_agent.n_obs,
                                 learning_agent.n_mem_states)
learning_agent.set_memory(mem_probs, logits=False)
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)

# Value stuff
def get_start_obs_value(value_fn, mdp):
    return (value_fn @ (mdp.p0 @ mdp.phi)).item()

lstd_v0, lstd_q0 = lstdq_lambda(pi_aug, mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(pi_aug, mem_aug_mdp, lambda_=args.lambda1)
start_value = get_start_obs_value(lstd_v1, mem_aug_mdp)
discrep_loss(pi_aug, mem_aug_mdp)

# Search stuff
info = learning_agent.optimize_memory(args.n_memory_trials)
if learning_agent.mem_optimizer == "fifo-queue":
    M = learning_agent.n_mem_states
    O = learning_agent.n_obs
    A = learning_agent.n_actions
    n_mem_fns = M**(M * O * A)
    print(f'Total memory functions: {n_mem_fns}')
    print(f'Number evaluated: {info["n_evals"]}')

plt.plot(range(len(info['discreps'])), info['discreps'])
plt.show()

# Search results stuff
print()
print(f'Best node: \n'
      f'{learning_agent.memory_probs}\n')
print(f'Best discrep: {info["best_discrep"]}')

# Final performance stuff
learning_agent.reset_policy()
planning_agent.pi_params = learning_agent.policy_logits
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)
pi_improvement(planning_agent, mem_aug_mdp, iterations=100, lr=0.1)
learning_agent.set_policy(planning_agent.pi_params, logits=True)

lstd_v0, lstd_q0 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda1)

end_value = get_start_obs_value(lstd_v1, mem_aug_mdp)
print(f'Start value: {start_value}')
print(f'End value: {end_value}')

# Save stuff
results = {
    'start_value': start_value,
    'end_value': end_value,
}
results.update(info)

dirname = f'results/discrete/{args.env}/{args.trial_id}'
json_filename = dirname + '/discrete_oracle.json'
npy_filename = dirname + '/memory.npy'
os.makedirs(dirname, exist_ok=True)
with open(json_filename, 'w') as f:
    json.dump(results, f)

np.save(npy_filename, learning_agent.memory_probs)
