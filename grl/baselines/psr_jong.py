"""
Implementing PSR modeling as in "Learning Predictive State Representations" by Nick Jong et al. (2003)
https://www.aaai.org/Papers/ICML/2003/ICML03-093.pdf

With discovery algorithm from "Predictive Representations of State" (Littman et al. 2001) 
https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf
"""

import numpy as np
import sympy as sym


def discover_tests(pomdp):
    """
    Discovers a set of core tests from a POMDP using algorithm from Littman et al. 2001.
    Modified to be BFS instead of DFS as in Littman, Jong et al 2003.
    Input: AbstractMDP (AbstractMDP? TODO)
    Output: Q, matrix of core tests
    """

    # let's assume for now that we can get a set of O matrices from the pomdp
    # we may need another routine to do that

    # A Test is a list of (action, observation) tuples, e.g [(0, 1), (1, 1)]. Null test is [].

    # grab stuff from POMDP for convenience & to match namespace of Littman et al. 2001.
    n = pomdp.n_states
    T = pomdp.T
    # Slight modification: Phi is a matrix of (States x obs) giving probability of each observation for each state.
    # we want to generate (obs x states x states) tensor, where each (states x states) matrix is diagonal, with diagonal value
    # i, i of each matrix being probability of that observation at state i.
    O = np.array([np.diag(v) for v in np.transpose(pomdp.phi)])
    # print(pomdp.phi)
    # print(O)


    # define u-function recursively as in Littman et al. 2001

    def _u_function(test):
        # recursive function that converts a test to a 1xn vec for independence testing
        if len(test) == 0:
            return np.ones((n,))
        else:
            prefix = test[0]
            suffix = test[1:]
            a = prefix[0]
            o = prefix[1]
            # modification: observation/state probabilities are independent of action
            return np.transpose(T[a] @ O[o] @ np.transpose(_u_function(suffix)))

    # def _u_function_one(ao_pair, vec):
    #   # function that performs only one step of the u-function: just append one action-observation pair.
    #   # this is more efficient than doing the full recursive computation each time, 
    #   # but requires that we've done that at least once (for the vec).
    #   a = ao_pair[0]
    #   o = ao_pair[1]
    #   return np.transpose(T[a] @ O[a][o] @ np.transpose(vec))

    # define BFS for core tests
    
    # start by visiting the null test
    to_visit = [[]]
    # start having visited nothing
    visited = []
    # efficiency buffer for u vectors
    visited_vectors = []

    # set of tests to return
    Q = []
    Q_uvectors = []

    # visit the root
    current = to_visit[0]
    to_visit = to_visit[1:]

    visited.append(current)
    visited_vectors.append(_u_function(current))
    #first one is always linearly independent of nothing
    Q.append(current)
    Q_uvectors.append(_u_function(current))
    # visit all 1-length tests
    for action in range(pomdp.n_actions):
        for obs in range(pomdp.n_obs):
            new_pair = (action, obs)
            # append test consisting of [(a', o'), (a, o), ...] to to_visit
            to_visit.append([new_pair] + current)


    #print(len(to_visit))
    while len(to_visit) > 0:
        # pop the next test
        current = to_visit[0]
        to_visit = to_visit[1:]

        visited.append(current)

        # generate a new u-vector
        current_u = _u_function(current)

        # check linear independence
        Q_check_m = np.stack(Q_uvectors + [current_u], axis=0)
        #print(current_u)

        _, indexes = sym.Matrix(Q_check_m).T.rref()  # T is for transpose
        if len(indexes) == len(Q_check_m):
            # all vectors are linearly independent, so we've found a new core test
            Q.append(current)
            Q_uvectors.append(current_u)

            # add all neighbors to visit queue
            for action in range(pomdp.n_actions):
                for obs in range(pomdp.n_obs):
                    new_pair = (action, obs)
                    # append test consisting of [(a', o'), (a, o), ...] to to_visit
                    to_visit.append([new_pair] + current)

    # when we're done, return Q
    return Q

class PSR:
    def __init__(self, pomdp, Q, pi):
        self.num_Q = len(Q)
        self.pi = pi
        self.Q = Q
        self.pred_vec = np.ones((self.num_Q,))

        # set of extension tests and associated weight vectors
        # currently zero-initalize? TODO
        self.weights = {}
        for action in range(pomdp.n_actions):
            for obs in range(pomdp.n_obs): 
                extension_pair = (action, obs)
                for core_test in Q:
                    ext_test = [extension_pair] + core_test
                    self.weights[tuple(ext_test)] = np.zeros((self.num_Q,))

    def get_pair_prob(self, action, observation):
        """
        Get the probability that the given observation followed the given action using the prediction vector.
        Make sure you have updated the prediction vector according to the history first!
        """
        tup = ((action, observation),)
        return np.transpose(self.pred_vec).dot(self.weights[tup])


    def recalculate_prediction_vector(self, h):
        """
        Re-Calculates prediction vector p(Q|h) for a given history h.
        A History is a list of (action, observation) pairs.
        """
        self.pred_vec = np.ones((self.num_Q,))

        for step in h:
            self.update_prediction_vector(step)


        return self.pred_vec

    def update_prediction_vector(self, pair):
        """
        Updates stored prediction vector with the given action-observation pair (tuple).
        returns nothing.
        """
        base_test_weight = self.weights[(pair,)]
        for i in range(self.num_Q):
            ext_test_weight = self.weights[tuple([pair] + self.Q[i])]
            self.pred_vec[i] = (np.transpose(self.pred_vec).dot(ext_test_weight)) / (np.transpose(self.pred_vec).dot(base_test_weight))
            # bound as in original paper
            self.pred_vec[i] = 1.0 if self.pred_vec[i] > 1.0 else self.pred_vec[i]
            self.pred_vec[i] = 0.0001 if self.pred_vec[i] < 0.0001 else self.pred_vec[i]


    def update_weights(self, history, stepsize):
        # get all sub-histories using list comprehension, ignoring duplicates
        # convert sub-histories to tuples for compatability with weight keys
        sub_histories = set([tuple(history[i: j]) for i in range(len(history)) for j in range(i + 1, len(history) + 1)])
        sub_histories_actions = set([tuple([tup[0] for tup in sh]) for sh in sub_histories])

        for ext_test in self.weights.keys():
            # always 1 in original paper
            importance_sampling_weight = 1
            test_actions = tuple([tup[0] for tup in ext_test])
            # check if the test's actions have ever been executed
            if test_actions in sub_histories_actions:
                # check if test observations were seen
                X_xt = 1 if ext_test in sub_histories else 0
                # run weight update
                self.weights[ext_test] = self.weights[ext_test] + stepsize * importance_sampling_weight * \
                (X_xt - np.transpose(self.pred_vec).dot(self.weights[ext_test])) * self.pred_vec









def learn_weights(pomdp, Q, pi=None, steps=10000000, start_stepsize=0.1, end_stepsize=0.00001, stepsize_delta=0.5, stepsize_reduce_interval=0.01):
    """
    Learning algorithm from Littman, Jong et al. 2003.
    Requires that a set of core tests Q have already been discovered (i.e. using discover_tests())
    A "test" is a list (or n-tuple) of (action, observation) pairs.

    Returns: PSR model instance, average 1-step error. TODO?
    """

    # grab stuff from POMDP for convenience & to match namespace of Littman et al. 2001.
    n = pomdp.n_states
    T = pomdp.T
    # Slight modification: Phi is a matrix of (States x obs) giving probability of each observation for each state.
    # we want to generate (obs x states x states) tensor, where each (states x states) matrix is diagonal, with diagonal value
    # i, i of each matrix being probability of that observation at state i.
    O = np.array([np.diag(v) for v in np.transpose(pomdp.phi)])

    # if we aren't given a specific policy to follow, choose actions uniformly at random
    if pi is None:
        pi = np.array([[1.0 / pomdp.n_actions for _ in range(pomdp.n_actions)] for _ in range(pomdp.n_obs)])

    # initialize model parameters
    model = PSR(pomdp, Q, pi)
    print(model.weights.keys())

    # get the longest extension test
    longest_test_len = 0
    for test in Q:
        if len(test) > longest_test_len:
            longest_test_len == len(test)

    longest_test_len = longest_test_len + 1


    # start training
    stepsize = start_stepsize
    training = False
    done = True
    history = []

    avg_error = 0.0

    for t in range(steps):
        if len(history) > 100000:
            # truncate history to last 100k steps to save memory
            history = history[1:]

        if done:
            # initialize state and get initial observation
            s = np.random.choice(pomdp.n_states, p=pomdp.p0)
            ob = pomdp.observe(s)
            done = False
        if training:


            # update weights and prediction vector
            model.update_weights(history, stepsize)
            model.update_prediction_vector(history[-1])


        if t >= longest_test_len and not training:
            # initialize weights
            model.update_weights(history, stepsize)

            # initialize core test vector
            model.recalculate_prediction_vector(history)
            training = True


        a = np.random.choice(pomdp.n_actions, p=pi[ob])
        next_s, r, done = pomdp.step(s, a, pomdp.gamma)
        ob = pomdp.observe(next_s)
        # TODO currently doing nothing with the reward? This is true of the original PSR algorithm 
        # but seems wrong if we want to do planning eventually.
        pair = (a, ob)
        history.append(pair)

        # adjust stepsize if needed
        if t != 0 and (t % (steps * stepsize_reduce_interval) == 0):
            print(f"Step {t}: Current average error is {avg_error}")
            stepsize = stepsize * stepsize_delta
            if stepsize < end_stepsize:
                stepsize = end_stepsize



        # Calculate error before we forget about which state we were just in
        step_tot_err = 0.0
        for observation in range(pomdp.n_obs):
            # get estimated and true probability that taking our action in the state results in this observation
            est = model.get_pair_prob(a, observation)
            # to calculate actual: get transition probabilities for current state/action, 
            # then calculate (p(next state) * p(observation))
            T_a_s = pomdp.T[a][s]
            true = 0.0
            for state in range(pomdp.n_states):
                # probability of transiting into a state times probability of yielding observation from that state
                true = true + (T_a_s[state] * pomdp.phi[state][observation])

            diff = true - est
            step_tot_err = step_tot_err + (diff * diff)

        # update running average
        avg_error = (avg_error * t + step_tot_err) / (t + 1)


        # advance state
        s = next_s

    
    return model, None














