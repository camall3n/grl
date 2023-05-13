import hashlib

import numpy as np

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

    def modify(self, action, obs, mem_op):
        new_probs = self.mem_probs.copy()
        new_probs[action, obs] = mem_op
        return SearchNode(new_probs)

    def modify_rowcol(self, action, obs, mem_row, mem_col):
        new_probs = self.mem_probs.copy()
        new_row = np.zeros((1, self.n_mem_states))
        new_row[0, mem_col] = 1
        new_probs[action, obs][mem_row] = new_row
        return SearchNode(new_probs)

    def get_successors(self, skip_hashes=set()):
        skip_hashes.add(self.mem_hash)
        n_actions, n_obs, _, _ = self.mem_probs.shape
        successors = []
        for a in range(n_actions):
            for o in range(n_obs):
                for mem_op in self.mem_ops:
                    successor = self.modify(a, o, mem_op)
                    if successor.mem_hash not in skip_hashes:
                        successors.append(successor)
        return successors

    def get_random_successor(self, modify_row=True):
        n_actions, n_obs, _, _ = self.mem_probs.shape
        a = np.random.choice(n_actions)
        o = np.random.choice(n_obs)

        if modify_row:
            mem_row = np.random.choice(self.n_mem_states)
            mem_col = np.random.choice(self.n_mem_states)
            successor = self.modify_rowcol(a, o, mem_row, mem_col)
        else:
            # for the original modify
            mem_op_idx = np.random.choice(range(len(self.mem_ops)))
            mem_op = self.mem_ops[mem_op_idx]
            successor = self.modify(a, o, mem_op)
        return successor
