import glob
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = []
for results_dir in sorted(glob.glob('results/sample_based/compare_sample_and_plan_03/*')):
    filepath = results_dir + '/sample_vs_plan.pkl'
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'rb') as file:
        info = pickle.load(file)

        n_samples = info['args']['replay_buffer_size']
        trial_id = info['args']['trial_id']

        lambda0 = info['args']['lambda0']
        lstd_q0 = info['lstd_q0']
        samp_q0 = info['samp_q0']
        value_error_0 = np.array((lstd_q0 - samp_q0)**2).mean()
        data.append({
            'n_samples': n_samples,
            'trial_id': trial_id,
            'lambda': lambda0,
            'value_error': value_error_0,
            'learner': 'active',
        })

        lambda1 = info['args']['lambda1']
        lstd_q1 = info['lstd_q1']
        samp_q1 = info['samp_q1']
        value_error_1 = np.array((lstd_q1 - samp_q1)**2).mean()
        data.append({
            'n_samples': n_samples,
            'trial_id': trial_id,
            'lambda': lambda1,
            'value_error': value_error_1,
            'learner': 'passive',
        })

df = pd.DataFrame(data)

#%%
sns.lineplot(data=df.query('n_samples>=1e6'), x='lambda', y='value_error', hue='n_samples', palette='colorblind')
plt.ylabel('Q-function MSE')
plt.xlabel('Lambda')
