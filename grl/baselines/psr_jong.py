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

	print(n)
	print(O.shape)


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
	# 	# function that performs only one step of the u-function: just append one action-observation pair.
	# 	# this is more efficient than doing the full recursive computation each time, 
	# 	# but requires that we've done that at least once (for the vec).
	# 	a = ao_pair[0]
	# 	o = ao_pair[1]
	# 	return np.transpose(T[a] @ O[a][o] @ np.transpose(vec))

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


	print(len(to_visit))
	while len(to_visit) > 0:
		# pop the next test
		current = to_visit[0]
		to_visit = to_visit[1:]

		visited.append(current)

		# generate a new u-vector
		current_u = _u_function(current)

		# check linear independence
		Q_check_m = np.stack(Q_uvectors + [current_u], axis=0)
		print(current_u)

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


def learn_test_probs(pomdp, Q):
	"""
	Learning algorithm from Littman, Jong et al. 2003.
	Requires that a set of core tests Q have already been discovered (i.e. using discover_tests())
	A "test" is a list of (action, observation) pairs.
	"""
	pass














