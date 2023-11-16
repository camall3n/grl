import hashlib

import numpy as np

from grl.utils.math import one_hot

def generate_hold_mem_fn(n_actions, n_obs, n_mem_states):
    HOLD = np.eye(n_mem_states)
    mem_probs = np.tile(HOLD[None, None, :, :], (n_actions, n_obs, 1, 1))
    return mem_probs

class SearchNode:
    def __init__(self, mem_probs):
        self.mem_probs = mem_probs
        self.mem_hash = self.round_and_hash(mem_probs)

        HOLD = np.eye(self.n_mem_states)
        TOGGLE = 1 - HOLD
        SET = np.concatenate((np.zeros((self.n_mem_states, 1)), np.ones((self.n_mem_states, 1))),
                             -1)
        RESET = 1 - SET
        self.mem_ops = [HOLD, TOGGLE, SET, RESET]

    @property
    def n_mem_states(self):
        return self.mem_probs.shape[-1]

    def round_and_hash(self, x, precision=0):
        x_bytes = np.round(x, precision).tobytes()
        hash_obj = hashlib.sha256(x_bytes)
        return hash_obj.hexdigest()

    def __hash__(self):
        return self.mem_hash

    def modify(self, action, obs, m, next_m):
        n_actions, n_obs, n_mem_states, _ = self.mem_probs.shape
        new_probs = self.mem_probs.copy()
        if action == n_actions:
            action = slice(None) # update every action
        if obs == n_obs:
            obs = slice(None) # update every observation
        if m == n_mem_states:
            m = slice(None) # update every memory state
        to_update = new_probs[action, obs, m]
        next_m_probs = one_hot(next_m, n=n_mem_states)
        new_probs[action, obs, m] = np.ones_like(to_update) * next_m_probs
        return SearchNode(new_probs)

    def get_successors(self, skip_hashes=set()):
        skip_hashes.add(self.mem_hash)
        successor_hashes = set()
        n_actions, n_obs, n_mem_states, _ = self.mem_probs.shape
        successors = []
        for a in range(n_actions + 1): # consider each single action as well as ALL actions
            for o in range(n_obs + 1): # consider each single obs as well as ALL observations
                for m in range(n_mem_states + 1):
                    for next_m in range(n_mem_states):
                        successor = self.modify(a, o, m, next_m)
                        if successor.mem_hash not in skip_hashes.union(successor_hashes):
                            successors.append(successor)
                            successor_hashes.add(successor.mem_hash)
        return successors

    def get_random_successor(self):
        n_actions, n_obs, n_mem_states, _ = self.mem_probs.shape
        a = np.random.choice(n_actions + 1) # consider each single action as well as ALL actions
        o = np.random.choice(n_obs + 1) # consider each single obs as well as ALL observations
        m = np.random.choice(n_mem_states + 1) # consider each single mem state as well as ALL mem states
        next_m = np.random.choice(n_mem_states) # next mem state must be a valid mem state
        successor = self.modify(a, o, m, next_m)
        return successor
