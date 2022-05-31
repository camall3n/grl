import numpy as np

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
        f = open(f'grl/environment/pomdp_files/{filename}.POMDP', 'r')
        self.contents = [
            x.strip() for x in f.readlines()
            if (not (x.startswith("#") or x.isspace()))
        ]

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
                    self.Z = np.zeros((len(self.actions),len(self.states), len(self.observations)))
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
            self.start = 1/n_states * np.ones(n_states)

        # cleanup
        f.close()

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
        action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            prob = float(pieces[3])
            self.T[action, start_state, next_state] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.T[action, start_state, next_state] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            if '*' in pieces[1]:
                start_state = slice(None)
            else:
                start_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.states)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.T[action, start_state, j] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
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
                    next_line = self.contents[i+2+j]
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
            action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            if '*' in pieces[1]:
                next_state = slice(None)
            else:
                next_state = self.states.index(pieces[1])
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
                next_state = self.states.index(pieces[1])
            if '*' in pieces[2]:
                obs = slice(None)
            else:
                obs = self.observations.index(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.Z[action, next_state, obs] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            if '*' in pieces[1]:
                next_state = slice(None)
            else:
                next_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.Z[action, next_state, j] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
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
                    next_line = self.contents[i+2+j]
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
            action = self.actions.index(pieces[0])

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
            self.__reward_ss(
                action, start_state_raw, next_state_raw, obs_raw, prob)
            return i + 1 if len(pieces) == 5 else i + 2
        elif len(pieces == 3):
            # case 2: R: <action> : <start-state> : <next-state>
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.R[action, start_state, next_state, j] = prob
            return i + 2
        elif len(pieces == 2):
            # case 3: R: <action> : <start-state>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.R[action, start_state, j, k] = prob
                next_line = self.contents[i+2+j]
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
            start_state = self.states.index(start_state_raw)
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
            next_state = self.states.index(next_state_raw)
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

        for _ in range(len(self.states) - 1):
            i += 1
            line = self.contents[i].split()
            components.append(np.array(line).astype('float'))

        pi = np.vstack(components)
        self.Pi_phi.append(pi)

        return i + 1

    def get_spec(self):
        return to_dict(self.T, self.R, self.discount, self.start, self.Z, self.Pi_phi)

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
