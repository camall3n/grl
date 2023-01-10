import numpy as np
from typing import List, Union

from .examples_lib import to_dict

class POMDPFile:
    """
    Adapted from https://github.com/mbforbes/py-pomdp/blob/master/pomdp.py

    Library of .POMDP files: http://pomdp.org/examples/
    For more info on format: http://pomdp.org/code/pomdp-file-spec.html
    """
    def __init__(self, filename):
        """
        Parses .pomdp file and loads info into this object's fields.
        Attributes:
            discount
            values
            states
            actions
            observations
            T
            Z
            R
            Pi_phi
        """
        f = open(filename, 'r')
        self.contents = []
        for x in f.readlines():
            if (not (x.startswith("#") or x.isspace())):
                # remove comments after lines too
                try:
                    hashtag_idx = x.index('#')
                    x = x[:hashtag_idx]
                except ValueError:
                    pass
                self.contents.append(x.strip())

        self.T = None
        self.Z = None
        self.R = None
        self.start = None
        self.Pi_phi = None

        # go through line by line
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            if line.startswith('discount'):
                i = self.__get_discount(i)
            elif line.startswith('values'):
                i = self.__get_value(i)
            elif line.startswith('states'):
                i = self.__get_states(i)
            elif line.startswith('actions'):
                i = self.__get_actions(i)
            elif line.startswith('observations'):
                i = self.__get_observations(i)
            elif line.startswith('start'):
                i = self.__get_start(i)
            elif line.startswith('T'):
                if self.T is None:
                    self.T = np.zeros((len(self.actions), len(self.states), len(self.states)))
                i = self.__get_transition(i)
            elif line.startswith('O'):
                if self.Z is None:
                    self.Z = np.zeros(
                        (len(self.actions), len(self.states), len(self.observations)))
                i = self.__get_observation(i)
            elif line.startswith('R'):
                if self.R is None:
                    self.R = np.zeros((len(self.actions), len(self.states), len(self.states)))
                i = self.__get_reward(i)
            elif line.startswith('Pi_phi'):
                if self.Pi_phi is None:
                    self.Pi_phi = []
                i = self.__get_pi_phi(i)
            else:
                raise Exception("Unrecognized line: " + line)

        # Default to uniform distribution over starting states
        if self.start is None:
            n_states = len(self.T[0])
            self.start = 1 / n_states * np.ones(n_states)

        # cleanup
        f.close()

    @staticmethod
    def get_indices(l: list, el: str) -> Union[List[int], int]:
        if el == '*':
            return list(range(len(l)))
        else:
            return l.index(el)

    def __get_discount(self, i):
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_value(self, i):
        # Currently just supports "values: reward". I.e. currently
        # meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __get_states(self, i):
        line = self.contents[i]
        self.states = line.split()[1:]
        if is_numeric(self.states):
            no_states = int(self.states[0])
            self.states = [str(x) for x in range(no_states)]
        return i + 1

    def __get_actions(self, i):
        line = self.contents[i]
        self.actions = line.split()[1:]
        if is_numeric(self.actions):
            no_actions = int(self.actions[0])
            self.actions = [str(x) for x in range(no_actions)]
        return i + 1

    def __get_observations(self, i):
        line = self.contents[i]
        self.observations = line.split()[1:]
        if is_numeric(self.observations):
            no_observations = int(self.observations[0])
            self.observations = [str(x) for x in range(no_observations)]
        return i + 1

    def __get_start(self, i):
        # TODO: handle other formats for this keyword
        line = self.contents[i]

        # Check if values are on this line or the next line
        if len(line.split()) == 1:
            i += 1
            line = self.contents[i].split()
        else:
            line = line.split()[1:]

        self.start = np.array(line).astype('float')

        return i + 1

    def __get_transition(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.get_indices(self.actions, pieces[0])

        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state = self.get_indices(self.states, pieces[1])
            next_state = self.get_indices(self.states, pieces[2])
            prob = float(pieces[3])
            self.T[action, start_state, next_state] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state = self.get_indices(self.states, pieces[1])
            next_state = self.get_indices(self.states, pieces[2])
            next_line = self.contents[i + 1]
            prob = float(next_line)
            self.T[action, start_state, next_state] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            if '*' in pieces[1]:
                start_state = slice(None)
            else:
                start_state = self.get_indices(self.states, pieces[1])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.states)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.T[action, start_state, j] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i + 1]
            if next_line == "identity":
                # case 4: T: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        prob = 1.0 if j == k else 0.0
                        self.T[action, j, k] = prob
                return i + 2
            elif next_line == "uniform":
                # case 5: T: <action>
                # uniform
                prob = 1.0 / float(len(self.states))
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        self.T[action, j, k] = prob
                return i + 2
            else:
                # case 6: T: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.states)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.T[action, j, k] = prob
                    next_line = self.contents[i + 2 + j]
                return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line " + line)

    def __get_observation(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        if pieces[0] == "*":
            # Case when action does not affect observation
            action = slice(None)
        else:
            action = self.get_indices(self.actions, pieces[0])

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            if '*' in pieces[1]:
                next_state = slice(None)
            else:
                next_state = self.get_indices(self.states, pieces[1])
            if '*' in pieces[2]:
                obs = slice(None)
            else:
                obs = self.observations.index(pieces[2])
            prob = float(pieces[3])
            self.Z[action, next_state, obs] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: O: <action> : <next-state> : <obs>
            # %f
            if '*' in pieces[1]:
                next_state = slice(None)
            else:
                next_state = self.get_indices(self.states, pieces[1])
            if '*' in pieces[2]:
                obs = slice(None)
            else:
                obs = self.observations.index(pieces[2])
            next_line = self.contents[i + 1]
            prob = float(next_line)
            self.Z[action, next_state, obs] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            if '*' in pieces[1]:
                next_state = slice(None)
            else:
                next_state = self.get_indices(self.states, pieces[1])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.Z[action, next_state, j] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i + 1]
            if next_line == "identity":
                # case 4: O: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        prob = 1.0 if j == k else 0.0
                        self.Z[action, j, k] = prob
                return i + 2
            elif next_line == "uniform":
                # case 5: O: <action>
                # uniform
                prob = 1.0 / float(len(self.observations))
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        self.Z[action, j, k] = prob
                return i + 2
            else:
                # case 6: O: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.observations)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.Z[action, j, k] = prob
                    next_line = self.contents[i + 2 + j]
                return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __get_reward(self, i):
        """
        Wild card * are allowed when specifying a single reward
        probability. They are not allowed when specifying a vector or
        matrix of probabilities.
        """
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        if pieces[0] == "*":
            action = slice(None)
        else:
            action = self.get_indices(self.actions, pieces[0])

        if len(pieces) == 5 or len(pieces) == 4:
            # case 1:
            # R: <action> : <start-state> : <next-state> : <obs> %f
            # any of <start-state>, <next-state>, and <obs> can be *
            # %f can be on the next line (case where len(pieces) == 4)
            start_state_raw = pieces[1]
            next_state_raw = pieces[2]
            obs_raw = pieces[3]
            prob = float(pieces[4]) if len(pieces) == 5 \
                else float(self.contents[i+1])
            self.__reward_ss(action, start_state_raw, next_state_raw, obs_raw, prob)
            return i + 1 if len(pieces) == 5 else i + 2
        elif len(pieces) == 3:
            # case 2: R: <action> : <start-state> : <next-state>
            # %f %f ... %f
            start_state = self.get_indices(self.states, pieces[1])
            next_state = self.get_indices(self.states, pieces[2])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.R[action, start_state, next_state, j] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: R: <action> : <start-state>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            start_state = self.get_indices(self.states, pieces[1])
            next_line = self.contents[i + 1]
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.R[action, start_state, j, k] = prob
                next_line = self.contents[i + 2 + j]
            return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __reward_ss(self, a, start_state_raw, next_state_raw, obs_raw, prob):
        """
        reward_ss means we're at the start state of the unrolling of the
        reward expression. start_state_raw could be * or the name of the
        real start state.
        """
        if start_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ns(a, i, next_state_raw, obs_raw, prob)
        else:
            start_state = self.get_indices(self.states, start_state_raw)
            self.__reward_ns(a, start_state, next_state_raw, obs_raw, prob)

    def __reward_ns(self, a, start_state, next_state_raw, obs_raw, prob):
        """
        reward_ns means we're at the next state of the unrolling of the
        reward expression. start_state is the number of the real start
        state, and next_state_raw could be * or the name of the real
        next state.
        """
        if next_state_raw == '*':
            self.R[a, start_state, :] = prob
        else:
            next_state = self.get_indices(self.states, next_state_raw)
            self.R[a, start_state, next_state] = prob

    def __get_pi_phi(self, i):
        components = []

        line = self.contents[i]

        # Check if first values are on this line or the next line
        if len(line.split()) == 1:
            i += 1
            line = self.contents[i].split()
        else:
            line = line.split()[1:]
        components.append(np.array(line).astype('float'))

        for _ in range(len(self.observations) - 1):
            i += 1
            line = self.contents[i].split()
            components.append(np.array(line).astype('float'))

        pi = np.vstack(components)
        self.Pi_phi.append(pi)

        return i + 1

    def convert_obs_actions(self):
        """
        If we're here, we want to convert our POMDP phi-action tensor
        into a phi tensor (so they're all the same).

        To do so, we add an extra "initialization" observation and state.
        We also need to expand our state space to include the most recent action.

        ONE BIG ASSUMPTION: In the POMDPs with phi(s, a), the action
        is applied to the state BEFORE an observation comes out.

        IN other words, if phi(obs|state,action), there are two possible conventions:
            1. action -> state -> obs
            2. state -> action -> obs
        We assume convention 1.
        """
        # first we construct our new transition function.
        # we go from |A| x |S| x |S| -> |A| x (|S| + 1)|A| x (|S| + 1)|A|
        # states are ordered as a0s0, a1s0, a2s0, ..., a0s1, a1s1, ..., etc.
        # i.e. all the actions that brought you to s0 appear first, then all the actions that brought you to s1, ...
        n_actions = self.T.shape[0]
        og_n_states = self.T.shape[1]

        # Create a new (initial) state and concatenate it to T such that all the actions that "precede" the initial
        # state lead to the same original start state distribution.

        # T_extra_start is |A| x (|S| + 1) x (|S| + 1)
        start_expanded = np.expand_dims(np.expand_dims(self.start, 0),
                                        0).repeat(n_actions, axis=0) # (A, 1, S)
        T_extra_start = np.concatenate((self.T, start_expanded), axis=1) # (A, S+1, S)
        extra_start_n_states = T_extra_start.shape[1]

        # But none of the newly added states produce rewards
        R_extra_start = np.concatenate((self.R, np.zeros_like(start_expanded)), axis=1)

        cannot_transition_to_s0 = np.zeros((n_actions, extra_start_n_states, 1)) # (A, S+1, S+1)
        no_rewards_to_s0 = np.zeros_like(cannot_transition_to_s0)
        T_extra_start = np.concatenate((T_extra_start, cannot_transition_to_s0),
                                       axis=2) # (A, S+1, S+1)
        R_extra_start = np.concatenate((R_extra_start, no_rewards_to_s0), axis=2)

        # Now we expand our T and R to add previous actions.
        # Initial state transition behavior does not depend on actions.
        # We start with the transition function.
        new_T = np.zeros(
            (n_actions, extra_start_n_states * n_actions, extra_start_n_states * n_actions))
        T_repeat_start_state = T_extra_start.repeat(n_actions, axis=1)
        for action in range(new_T.shape[0]):
            for action_state in range(new_T.shape[1]):
                # The "preceding" action of the "next" state needs to match the selected action,
                # so we copy these probabilities and leave the rest at zero.
                new_T[action, action_state, np.arange(extra_start_n_states) * n_actions + action] = \
                    T_repeat_start_state[action, action_state]
        """
        Now our reward function - it should just be our current reward
        function but repeated over actions.
        Again there are two conventions:
            1. Define rewards as being over only the valid transitions.
            2. Define rewards as being over all expressible transitions.
        Since we have carefully accounted for transition probabilities above, we opt for the second convention here.
        """
        new_R = R_extra_start.repeat(n_actions, axis=1).repeat(n_actions, axis=2)

        # Now the phi function. We have to add an observation for the new start state
        # as well as add a new row for the new start state

        # Create a new (initial) observation and concatenate it to phi such that
        # all the action-states lead to the same original observation distributions.
        cannot_see_new_obs = np.zeros((n_actions, og_n_states, 1))
        Z_extra_start_obs = np.concatenate((self.Z, cannot_see_new_obs), axis=2) # (A, S, O+1)
        extra_obs_n_obs = Z_extra_start_obs.shape[2]

        # New start state can only emit this new start obs.
        new_start_phi = np.zeros((n_actions, 1, extra_obs_n_obs))
        new_start_phi[:, 0, -1] = 1
        new_Z_actions_first = np.concatenate((Z_extra_start_obs, new_start_phi),
                                             axis=1) # (A, S+1, O+1)

        # This swapaxes keeps the correct convention for combining action-states.
        new_Z = np.swapaxes(new_Z_actions_first, 0,
                            1).reshape(-1, new_Z_actions_first.shape[2]) # ((S+1)*A, O+1)

        # We need to add a policy for our starting observation
        new_pi_phi = None
        if self.Pi_phi is not None:
            new_pi_phi = []
            # Initial-state policy is uniform random.
            uniform_dist = np.ones((1, n_actions)) / n_actions
            for pi in self.Pi_phi:
                # Otherwise it's the same.
                new_pi_phi.append(np.concatenate((pi, uniform_dist), axis=0))

        # We have a new start state - update start state dist.
        # Expand start states over actions as well.
        new_start = np.zeros(new_T.shape[-1])

        # Set the (last state, action) pairs as equal starting probabilities.
        # This is arbitrary, since all actions lead to the initial state, and all actions have the
        # same effect. But basically, start with the initial state we invented.
        new_start[-n_actions:] = (1 / n_actions)

        return to_dict(new_T, new_R, self.discount, new_start, new_Z, new_pi_phi)

    def get_spec(self):
        phi = self.Z[0]
        if len(self.Z) > 1:
            # first we test if all phi matrices are the same for all actions
            prev_z = self.Z[0]
            all_same = True
            for z in self.Z[1:]:
                all_same &= np.allclose(prev_z, z)
                if not all_same:
                    break

            if not all_same:
                return self.convert_obs_actions()

        return to_dict(self.T, self.R, self.discount, self.start, phi, self.Pi_phi)

    def print_summary(self):
        print("discount:", self.discount)
        print("values:", self.values)
        print("states:", self.states)
        print("actions:", self.actions)
        print("observations:", self.observations)
        print("")
        print("T:", self.T)
        print("")
        print("Z:", self.Z)
        print("")
        print("R:", self.R)

def is_numeric(lst):
    if len(lst) == 1:
        try:
            int(lst[0])
            return True
        except Exception:
            return False
    else:
        return False
