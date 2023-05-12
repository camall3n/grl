from collections import deque
import hashlib
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
from grl.memory import memory_cross_product
from scripts.learning_agent.memory_iteration import parse_args

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
    mellowmax_beta=10.,
    discrep_loss='mse',
    study_name='compare_sample_and_plan_04/' + args.study_name,
)

planning_agent = AnalyticalAgent(
    pi_params=learning_agent.policy_probs,
    rand_key=None,
    mem_params=learning_agent.memory_probs,
    value_type='q',
)

# Policy stuff
pi_improvement(planning_agent, env, lr=0.1)
learning_agent.set_policy(planning_agent.pi_params, logits=True)
learning_agent.add_memory()
pi_aug = learning_agent.policy_probs

# Memory stuff
HOLD = np.eye(learning_agent.n_mem_states)
TOGGLE = 1 - HOLD
SET = np.concatenate((np.zeros(
    (learning_agent.n_mem_states, 1)), np.ones((learning_agent.n_mem_states, 1))), -1)
RESET = 1 - SET

mem_probs = np.tile(HOLD[None, None, :, :], (learning_agent.n_actions, learning_agent.n_obs, 1, 1))
learning_agent.set_memory(mem_probs, logits=False)
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)

# Value stuff
lstd_v0, lstd_q0 = lstdq_lambda(pi_aug, mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(pi_aug, mem_aug_mdp, lambda_=args.lambda1)

def get_start_obs_value(value_fn, mdp):
    return (value_fn @ (mdp.p0 @ mdp.phi)).item()

start_value = get_start_obs_value(lstd_v0, mem_aug_mdp)
discrep_loss(pi_aug, mem_aug_mdp)

# Search stuff
class SearchNode:
    def __init__(self, mem_probs=None):
        if mem_probs is None:
            mem_probs = np.tile(HOLD[None, None, :, :],
                                (learning_agent.n_actions, learning_agent.n_obs, 1, 1))

        self.mem_probs = mem_probs
        self.mem_hash = self.round_and_hash(mem_probs)

    def round_and_hash(self, x, precision=0):
        x_bytes = np.round(x, precision).tobytes()
        hash_obj = hashlib.sha256(x_bytes)
        return hash_obj.hexdigest()

    def modify(self, action, obs, mem_op):
        new_probs = self.mem_probs.copy()
        new_probs[action, obs] = mem_op
        return SearchNode(new_probs)

    def get_successors(self, skip_hashes=set()):
        skip_hashes.add(self.mem_hash)
        n_actions, n_obs, _, _ = self.mem_probs.shape
        successors = []
        for a in range(n_actions):
            for o in range(n_obs):
                for mem_op in [HOLD, TOGGLE, SET, RESET]:
                    successor = self.modify(a, o, mem_op)
                    if successor.mem_hash not in skip_hashes:
                        successors.append(successor)
        return successors

    def evaluate(self):
        learning_agent.set_memory(self.mem_probs, logits=False)
        mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)
        return discrep_loss(learning_agent.policy_probs, mem_aug_mdp)

def optimize_memory(mem_probs):
    s = SearchNode(mem_probs)

    visited = set()
    frontier = deque([s])
    best_discrep = np.inf
    best_node = None
    n_evals = 0

    while frontier:
        node = frontier.popleft()
        discrep = node.evaluate()[0] + np.random.normal(loc=0, scale=0.00)
        n_evals += 1
        print(f'discrep = {discrep}')
        visited.add(node.mem_hash)

        if discrep < best_discrep:
            print(f'New best discrep: {discrep}')
            best_discrep = discrep
            best_node = node
            successors = node.get_successors(skip_hashes=visited)
            frontier.extend(successors)

    info = {
        'n_evals': n_evals,
        'best_discrep': best_discrep.item(),
    }
    return best_node, info

best_node, info = optimize_memory(mem_probs)

# Search results stuff
print()
print(f'Best node: \n'
      f'{best_node.mem_probs}\n')
M = learning_agent.n_mem_states
O = learning_agent.n_obs
A = learning_agent.n_actions
n_mem_fns = M**(M * O * A)
print(f'Total memory functions: {n_mem_fns}')
print(f'Number evaluated: {info["n_evals"]}')
print(f'Best discrep: {info["best_discrep"]}')

# Final performance stuff
learning_agent.set_memory(best_node.mem_probs, logits=False)
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)

learning_agent.reset_policy()
planning_agent.pi_params = learning_agent.policy_logits
pi_improvement(planning_agent, mem_aug_mdp, lr=0.1)
learning_agent.set_policy(planning_agent.pi_params, logits=True)

lstd_v0, lstd_q0 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda1)

end_value = get_start_obs_value(lstd_v0, mem_aug_mdp)
print(f'Start value: {start_value}')
print(f'End value: {end_value}')

# Save stuff
results = {
    'start_value': start_value,
    'end_value': end_value,
}
results.update(info)

dirname = f'results/discrete/{args.env}/{args.trial_id:03d}'
json_filename = dirname + '/discrete_oracle.json'
npy_filename = dirname + '/memory.npy'
os.makedirs(dirname, exist_ok=True)
with open(json_filename, 'w') as f:
    json.dump(results, f)

np.save(npy_filename, best_node.mem_probs)
