from collections import defaultdict
import glob
import os

import logging

# logging.basicConfig(format='%(message)s', level=logging.INFO)
# logging.getLogger().setLevel(logging.INFO)

import numpy as np

from grl.run import run_generated

# files = glob.glob('grl/results/1bit_mem_conjecture/1660936953-1433578/*')

# positive_results = [os.path.splitext(os.path.basename(f))[0].split('_') for f in files]
# n_mdps = max([int(r[0]) for r in positive_results]) + 1
# n_policies = max([int(r[1]) for r in positive_results]) + 1
# n_mem_fns = max([int(r[2]) for r in positive_results]) + 1

# markov_chains = defaultdict(list)
# for pomdp_id, policy_id, mem_fn_id in positive_results:
#     markov_chains[(int(pomdp_id), int(policy_id))].append(mem_fn_id)

# for pomdp_id in range(n_mdps):
#     for policy_id in range(n_policies):
#         if len(markov_chains[(pomdp_id, policy_id)]) == 0:
#             negative_results.append((pomdp_id, policy_id))

negative_results = [(6, 1), (9, 1)] #(4, 0)]#, (4, 1),

if len(negative_results) > 0:
    print('No mem fns found with pareto-improvement for:')
    for pomdp_id, policy_id in negative_results:
        print(f'POMDP: {pomdp_id}, π: {policy_id}')
        dir = f'grl/environment/pomdp_files/generated/1660936953-1433578'
        results = run_generated(dir, pomdp_id)

mem_fn_improved_discrep = np.asarray(results)
mem_fn_improved_discrep.any(axis=0)
