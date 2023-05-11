from collections import deque
from definitions import ROOT_DIR
import hashlib
import pickle

import numpy as np
from jax.config import config
from pathlib import Path

from grl.agents.actorcritic import ActorCritic
from grl.agents.analytical import AnalyticalAgent
from grl.memory_iteration import pi_improvement
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import lstdq_lambda
from grl.utils.loss import discrep_loss
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

spec = load_spec(spec_name,
                 memory_id=str(16),
                 corridor_length=corridor_length,
                 discount=discount,
                 junction_up_pi=junction_up_pi,
                 epsilon=epsilon)

mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
pi = spec['Pi_phi'][0]

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
    discrep_loss=error_type,
    study_name='compare_sample_and_plan_04/' + args.study_name,
)
learning_agent.set_policy(pi, logits=False)
learning_agent.add_memory()

HOLD = np.eye(learning_agent.n_mem_states)
TOGGLE = 1 - HOLD
SET = np.concatenate((np.zeros((learning_agent.n_mem_states, 1)), np.ones((learning_agent.n_mem_states, 1))), -1)
RESET = 1 - SET

mem_probs = np.tile(HOLD[None, None, :, :], (learning_agent.n_actions, learning_agent.n_obs, 1, 1))
learning_agent.set_memory(mem_probs, logits=False)
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)

lstd_v0, lstd_q0 = lstdq_lambda(pi.repeat(2, axis=0), mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(pi.repeat(2, axis=0), mem_aug_mdp, lambda_=args.lambda1)

aug_policy = learning_agent.policy_probs
discrep_loss(aug_policy, mem_aug_mdp)

class SearchNode:
    def __init__(self, mem_probs=None):
        if mem_probs is None:
            mem_probs = np.tile(HOLD[None, None, :, :], (learning_agent.n_actions, learning_agent.n_obs, 1, 1))

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

s = SearchNode(mem_probs)

visited = set()
frontier = deque([s])
best_discrep = np.inf
best_node = None

while frontier:
    node = frontier.popleft()
    discrep = node.evaluate()[0]
    print(f'discrep = {discrep}')
    visited.add(node.mem_hash)

    if discrep < best_discrep:
        print(f'New best discrep: {discrep}')
        best_discrep = discrep
        best_node = node
        successors = node.get_successors(skip_hashes=visited)
        frontier.extend(successors)

print(f'Best discrep: {best_discrep}')
print(f'Best node: \n'
      f'{best_node.mem_probs}\n'
      f'{best_node.evaluate()}')

discrep, mc, td = s.evaluate()
(mc['q'] - td['q']).round(2)

learning_agent.set_memory(best_node.mem_probs, logits=False)
mem_aug_mdp = memory_cross_product(learning_agent.memory_logits, env)
lstd_v0, lstd_q0 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda1)

planning_agent = AnalyticalAgent(learning_agent.policy_probs, None, mem_params=mem_probs, value_type='q')
pi_improvement(planning_agent, mem_aug_mdp, lr=0.1)
planning_agent.pi_params
learning_agent.set_policy(planning_agent.pi_params, logits=True)
learning_agent.policy_probs

lstd_v0, lstd_q0 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda0)
lstd_v1, lstd_q1 = lstdq_lambda(learning_agent.policy_probs, mem_aug_mdp, lambda_=args.lambda1)
