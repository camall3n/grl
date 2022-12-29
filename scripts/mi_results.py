#!/usr/bin/env python
# coding: utf-8

import numpy as np
from jax.nn import softmax
from pathlib import Path
from collections import namedtuple

np.set_printoptions(precision=4)

from grl.utils import load_info
from definitions import ROOT_DIR

# In[226]:

results_dir = Path(ROOT_DIR, 'results', 'runs_tmaze_dm')

split_by = ['spec', 'algo', 'mi_iterations', 'policy_optim_alg']
Args = namedtuple('args', split_by)

# In[239]:

all_results = {}

for results_path in results_dir.iterdir():
    if results_path.suffix != '.npy':
        continue

    info = load_info(results_path)
    args = info['args']
    logs = info['logs']
    agent = info['agent']

    single_res = {
        'final_mem_discrep_v': logs['final_mem_discrep']['v'],
        'final_mem_discrep_q': logs['final_mem_discrep']['q'],
        'final_mem': agent.memory,
        'final_policy': agent.policy,
        'initial_improvement_policy': logs['initial_improvement_policy'],
        'initial_expanded_improvement_policy': logs['initial_expanded_improvement_policy'],
        'initial_improvement_discrep': logs['initial_improvement_discrep'],
        'initial_discrep': logs['initial_discrep'],
        'initial_policy': logs['initial_policy'],
    }

    hparams = Args(*tuple(args[s] for s in split_by))

    if hparams not in all_results:
        all_results[hparams] = {}

    for k, v in single_res.items():
        if k not in all_results[hparams]:
            all_results[hparams][k] = []
        all_results[hparams][k].append(v)
    all_results[hparams]['args'] = args

for hparams, res_dict in all_results.items():
    for k, v in res_dict.items():
        all_results[hparams][k] = np.stack(v)

# In[240]:

all_results[list(all_results.keys())[0]]['final_mem'].shape

# In[241]:

logs['final_outputs']

# In[ ]:

# In[249]:

res = all_results[list(all_results.keys())[0]]
idx = 1
mem = res['final_mem'][idx]
final_policy = res['final_policy'][idx]
first_improvement_policy = res['initial_improvement_policy'][idx]

initial_policy = res['initial_policy'][idx]

initial_expanded_improvement_policy = res['initial_expanded_improvement_policy'][idx]
initial_improvement_discrep = res['initial_improvement_discrep'][idx]
initial_discrep = res['initial_discrep'][idx]
right = mem[2]
# print(f"Memory for RIGHT action:\n"
#       f"Goal(Up) start obs, from mem[0]: {right[0, 0]}\n"
#       f"Goal(Down) start obs, from mem[0]: {right[1, 0]}\n"
#       f"Corridor obs mem func:\n{right[2]}")
print()
print(f"initial policy: \n {initial_policy}")
print("initial policy lambda discreps:")
print(initial_discrep['v'])
print(initial_discrep['q'])
print()
print(f"policy after first improvement: \n{first_improvement_policy}")
print("lambda discreps:")
print(initial_improvement_discrep['v'])
print(initial_improvement_discrep['q'])
print("MC Q-vals of policy after first improvement:")
print(initial_improvement_discrep['mc_vals_q'])
print("TD Q-vals of policy after first improvement:")
print(initial_improvement_discrep['td_vals_q'])

# In[106]:

print(f"Junction policy for memory 0: {policy[3*2]}")
print(f"Junction policy for memory 1: {policy[3*2 + 1]}")

print(f"Policy after first improvement:\n {first_improvement_policy}")
print(f"Policy after initial expansion:\n {initial_expanded_improvement_policy}")

# In[118]:

res['initial_policy'].mean(axis=0)

# In[225]:

softmax(np.random.normal(size=res['initial_policy'][0].shape) * 0.2, axis=-1)

# In[ ]:
