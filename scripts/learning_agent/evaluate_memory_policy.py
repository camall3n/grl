import numpy as np

#%%
experiment_name = 'test-numpy-saving'
env_name = 'tmaze_2_two_thirds_up'
seed = '1'

results_dir = f'results/sample_based/{experiment_name}/{env_name}/{seed}/'
memory = np.load(results_dir + 'memory.npy')
policy = np.load(results_dir + 'policy.npy')
td = np.load(results_dir + 'q_td.npy')
mc = np.load(results_dir + 'q_mc.npy')
