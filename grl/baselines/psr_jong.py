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

    # set of tests to return
    Q = []
    Q_uvectors = []

    # visit the root
    current = to_visit[0]
    to_visit = to_visit[1:]

    visited.append(current)
    #don't add the null test to cores (parity with Jong code)
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
        self.__init_max = 2.0 / len(Q)
        self.__init_min = 0.0001
        self.__random_gen = np.random.default_rng(seed=10)


        self.num_Q = len(Q)
        self.pi = pi
        self.Q = Q
        self.pred_vec = self.__random_gen.uniform(self.__init_min, 1.0, (self.num_Q,))
        self._sub_histories = set()
        self._sub_histories_actions = set()


        # set of extension tests and associated weight vectors
        # currently zero-initalize? TODO
        self.weights = {}
        for action in range(pomdp.n_actions):
            for obs in range(pomdp.n_obs): 
                extension_pair = (action, obs)
                # add a null test extension
                self.weights[(extension_pair,)] = self.__random_gen.uniform(self.__init_min, self.__init_max, (self.num_Q,))

                for core_test in Q:
                    ext_test = [extension_pair] + core_test
                    # TODO random initialization? hard to find what original paper did
                    self.weights[tuple(ext_test)] = self.__random_gen.uniform(self.__init_min, self.__init_max, (self.num_Q,))

    def flush_history(self):
        # dump stored subhistories and learned prediction vectors.
        # Use this if you're restarting in a new environment; for instance, if you are
        # evaluating on a run separate from the training run.
        self.pred_vec = self.__random_gen.uniform(self.__init_min, 1.0, (self.num_Q,))
        self._sub_histories = set()
        self._sub_histories_actions = set()

    def get_pair_prob(self, action, observation):
        """
        Get the probability that the given observation followed the given action using the prediction vector.
        Make sure you have updated the prediction vector according to the history first!
        """
        tup = ((action, observation),)
        return np.transpose(self.pred_vec).dot(self.weights[tup])


    def recalculate_prediction_vector(self, h, start_idx, t):
        """
        Re-Calculates prediction vector p(Q|h) for a given history h.
        A History is a list of (action, observation) pairs.
        """
        self.pred_vec = self.__random_gen.uniform(self.__init_min, 1.0, (self.num_Q,))
        cap = len(h)
        end_idx = t % cap
        for i in range(cap):
            seq_idx = (start_idx + i) % cap
            if h[seq_idx] == (None, None) or seq_idx == (end_idx + 1) % cap:
                break
            self._update_prediction_vector(h[seq_idx])


        return self.pred_vec

    def _update_prediction_vector(self, pair):
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
            self.pred_vec[i] = self.__init_min if self.pred_vec[i] < self.__init_min else self.pred_vec[i]


    def update_weights(self, history, stepsize, t):
        # Update history according to which tests were executed at time t.
        # assumes that history is a looped queue (of size c) with at least t + k entries, where k is the length of the longest extension 
        # test. 
        # If t + k is greater than c, assumes that the history loops back around at index 0.

        cap = len(history)
        t_true = t % cap

        def _executed(test):
            # check if test was executed at time t.
            seq_idx = t_true
            for i in range(len(test)):
                seq_idx = (seq_idx + 1) % cap
                if test[i][0] != history[seq_idx][0]:
                    return False
            return True
        
        def _outcome(test):
            # check if test's observations were seen at time t.
            seq_idx = t_true
            for i in range(len(test)):
                seq_idx = (seq_idx + 1) % cap
                if test[i][1] != history[seq_idx][1]:
                    return False
                
            return True

            


        for ext_test in self.weights.keys():
            # always 1 in original paper
            importance_sampling_weight = 1
            # check if the test's actions have ever been executed
            if _executed(ext_test):
                # check if test observations were seen
                X_xt = 1 if _outcome(ext_test) else 0
                # run weight update
                self.weights[ext_test] = self.weights[ext_test] + stepsize * importance_sampling_weight * \
                (X_xt - np.transpose(self.pred_vec).dot(self.weights[ext_test])) * self.pred_vec









def learn_weights(pomdp, Q, pi=None, steps=10000000, start_stepsize=0.1, end_stepsize=0.00001, stepsize_delta=0.9, stepsize_reduce_interval=0.01, mem_cap = 1024):
    """
    Learning algorithm from Littman, Jong et al. 2003.
    Requires that a set of core tests Q have already been discovered (i.e. using discover_tests())
    A "test" is a list (or n-tuple) of (action, observation) pairs.

    Returns: PSR model instance, average 1-step error. TODO?
    """

    # grab stuff from POMDP for convenience & to match namespace of Littman et al. 2001.
    n = pomdp.n_states
   
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
            longest_test_len = len(test)

    longest_test_len = longest_test_len + 1


    # start training
    stepsize = start_stepsize
    training = False
    done = True
    # we're going to treat this like a C array even though it isn't, for parity with Jong's code
    history = [(None,None) for _ in range(mem_cap)]

    errors = []

    avg_error = 0.0
    tot_error = 0.0

    # keep track of the last few states we were in for calculating error
    s_buf = [None for _ in range(longest_test_len + 1)]

    for t in range(steps):
        if training:


            # update weights and prediction vector
            model.update_weights(history, stepsize, t)
            # switch to recalculating as that should be more correct
            model.recalculate_prediction_vector(history, start_idx, t)

        if done:
            #print("Finished epoch")
            # initialize state and get initial observation
            # also wipe history, because we started over
            s = np.random.choice(pomdp.n_states, p=pomdp.p0)
            ob = pomdp.observe(s)
            done = False
            training = False
            model.flush_history()
            history = [(None,None) for _ in range(mem_cap)]
            start_idx = t % mem_cap

            s_buf.append(s)
            s_buf = s_buf[1:]

            # run a number of fake training steps to generate a long enough history to learn from
            # (this is what they do in the original code)
            for i in range(longest_test_len):
                a = np.random.choice(pomdp.n_actions, p=pi[ob])
                next_s, r, done = pomdp.step(s, a, pomdp.gamma)
                ob = pomdp.observe(next_s)
                # TODO currently doing nothing with the reward? This is true of the original PSR algorithm 
                # but seems wrong if we want to do planning eventually.
                pair = (a, ob)
                ins_idx = (start_idx + i) % mem_cap
                history[ins_idx] = pair 

                
                s = next_s
                s_buf.append(s)
                s_buf = s_buf[1:]              

        if len(history) >= longest_test_len and not training:
            # initialize weights
            model.update_weights(history, stepsize, t)

            # initialize core test vector
            model.recalculate_prediction_vector(history, start_idx, t)
            training = True

        # here's the confusing thing:
        # at any given step t,
        # we are performing the action and gathering an observation for step (t + longest_test_len)
        # while updating the weights and calculating error for step t.

        a = np.random.choice(pomdp.n_actions, p=pi[ob])
        next_s, r, done = pomdp.step(s, a, pomdp.gamma)
        ob = pomdp.observe(next_s)
        # TODO currently doing nothing with the reward? This is true of the original PSR algorithm 
        # but seems wrong if we want to do planning eventually.
        pair = (a, ob)
        place_idx = (t + longest_test_len) % mem_cap
        history[place_idx] = pair
        if place_idx == start_idx:
            start_idx = (start_idx + 1) % mem_cap
        

        # adjust stepsize if needed
        if t != 0 and (t % (steps * stepsize_reduce_interval) == 0):
            print(f"Step {t} (training {training}): Current average error is {avg_error}")
            stepsize = stepsize * stepsize_delta
            if stepsize < end_stepsize:
                stepsize = end_stepsize



        # Calculate error before we forget about which state we were just in
        step_tot_err = 0.0
        for observation in range(pomdp.n_obs):
            # get estimated and true probability that taking our action in the state results in this observation
            est = model.get_pair_prob(a, observation)
            # to calculate actual: get transition probabilities for state/action at time t, 
            # then calculate (p(next state) * p(observation))
            t_state = s_buf[0]
            T_a_s = pomdp.T[a][t_state]
            true = 0.0
            for state in range(pomdp.n_states):
                # probability of transiting into a state times probability of yielding observation from that state
                true = true + (T_a_s[state] * pomdp.phi[state][observation])

            #print(f"Prev_state: {t_state} | True error: {true} | Estimated: {est}")

            diff = true - est
            step_tot_err = step_tot_err + (diff * diff)

        # update running average
        avg_error = (avg_error * t + step_tot_err) / (t + 1)
        tot_error = tot_error + step_tot_err

        # TODO error reporting for float/reset
        if (t + 1) % 10000 == 0:
            errors.append(tot_error / 10000)
            tot_error = 0.0
            print(f"Step {t} 10,000 evg error: {errors[-1]}")


        # advance state
        s = next_s
        s_buf.append(s)
        s_buf = s_buf[1:]

    # TODO
    return model, avg_error

