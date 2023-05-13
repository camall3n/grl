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

import math
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

start_value = get_start_obs_value(lstd_v1, mem_aug_mdp)
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

    def modify_rowcol(self, action, obs, mem_row, mem_col):
        new_probs = self.mem_probs.copy()
        new_row = np.zeros((1,learning_agent.n_mem_states))
        new_row[0,mem_col] = 1
        new_probs[action, obs][mem_row] = new_row
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
    
    def get_random_successor(self, modify_row=True):
        n_actions, n_obs, _, _ = self.mem_probs.shape
        a = np.random.choice(n_actions)
        o = np.random.choice(n_actions)

        if modify_row: 
            mem_row = np.random.choice(learning_agent.n_mem_states)
            mem_col = np.random.choice(learning_agent.n_mem_states)
            successor = self.modify_rowcol(a, o, mem_row, mem_col)
        else:
            # for the original modify
            mem_ops = [HOLD, TOGGLE, SET, RESET]
            mem_op_idx = np.random.choice(range(len(mem_ops)))
            mem_op = mem_ops[mem_op_idx]
        
        return successor

    def evaluate(self):
        learning_agent.set_memory(self.mem_probs, logits=False)
        mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)
        return discrep_loss(learning_agent.policy_probs, mem_aug_mdp)

def queue_search(mem_probs):
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
        # print(f'discrep = {discrep}')
        visited.add(node.mem_hash)

        if discrep < best_discrep:
            # print(f'New best discrep: {discrep}')
            best_discrep = discrep
            best_node = node
            successors = node.get_successors(skip_hashes=visited)
            frontier.extend(successors)

    info = {
        'n_evals': n_evals,
        'best_discrep': best_discrep.item(),
    }
    return best_node, info

def simulated_annealing(mem_probs, beta=1e3, cooling_rate=0.99, n=200):
    # simulated annealing
    # beta = 1/temp
    discs = []
    s = SearchNode(mem_probs)
    node = s
    p = node.evaluate()[0]
    best_node = s
    best_p = p
    for i in range(n):
        successor = node.get_random_successor()
        p2 = successor.evaluate()[0]
        de = p2 - p
        if p2 == p:
            accept = 0
        else:
            accept = math.exp(-de*beta)
        # decide whether to accept the transition
        if de <= 0:
            node = successor
            p = p2
            if p < best_p:
                best_p = p
                best_node = node
        elif np.random.random() < accept:
            node = successor
            p = p2
        if i % 1 == 0:
            discs.append(p)
        beta /= cooling_rate
    info = {
        'best_discrep': discs[-1].tolist(),
        'beta': beta,
        'cooling_rate': cooling_rate,
        'n': n,
    }
    return best_node, info, discs
    
mode = "queue"
assert mode in ["queue", "sa"], "invalid mode"

if mode == "queue":
    best_node, info = queue_search(mem_probs)
    M = learning_agent.n_mem_states
    O = learning_agent.n_obs
    A = learning_agent.n_actions
    n_mem_fns = M**(M * O * A)
    print(f'Total memory functions: {n_mem_fns}')
    print(f'Number evaluated: {info["n_evals"]}')
    
elif mode == "sa":
    beta = 1e3
    cooling_rate = 0.99
    n = 200
    best_node, info, discs = simulated_annealing(mem_probs, beta, cooling_rate, n)
    plt.plot(range(len(discs)), discs)
    plt.show()

# Search results stuff
print()
print(f'Best node: \n'
      f'{best_node.mem_probs}\n')
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

np.save(npy_filename, best_node.mem_probs)