import glob
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = []
for results_dir in sorted(glob.glob('results/sample_based/compare_sample_and_plan_04/*')):
    filepath = results_dir + '/sample_vs_plan.pkl'
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'rb') as file:
        info = pickle.load(file)

        n_samples = info['args']['replay_buffer_size']
        learning_rate = info['args']['learning_rate']
        trial_id = info['args']['trial_id']

        lambda0 = info['args']['lambda0']
        lstd_q0 = info['lstd_q0']
        samp_q0 = info['samp_q0']
        value_error_0 = np.array((lstd_q0 - samp_q0)**2).mean()

        lambda1 = info['args']['lambda1']
        lstd_q1 = info['lstd_q1']
        samp_q1 = info['samp_q1']
        value_error_1 = np.array((lstd_q1 - samp_q1)**2).mean()

        lstd_discrep = (np.array(lstd_q1 - lstd_q0)**2).mean()
        samp_discrep = (np.array(samp_q1 - samp_q0)**2).mean()

        data.append({
            'n_samples': n_samples,
            'learning_rate': learning_rate,
            'trial_id': trial_id,
            'lambda': lambda0,
            'lambda_mode': 'td',
            'lstd_q': lstd_q0,
            'samp_q': samp_q0,
            'value_error': value_error_0,
            'lstd_discrep': lstd_discrep,
            'samp_discrep': samp_discrep,
        })

        data.append({
            'n_samples': n_samples,
            'learning_rate': learning_rate,
            'trial_id': trial_id,
            'lambda': lambda1,
            'lambda_mode': 'mc',
            'lstd_q': lstd_q1,
            'samp_q': samp_q1,
            'value_error': value_error_1,
            'lstd_discrep': lstd_discrep,
            'samp_discrep': samp_discrep,
        })

df = pd.DataFrame(data)

#%%
sns.lineplot(data=df, x='n_samples', y='value_error', hue='learning_rate', style='lambda')
plt.hlines(df.lstd_discrep.mean(), df.n_samples.min(), df.n_samples.max(), linestyle='--', color='r')
plt.text(x=df.n_samples.min(), y=lstd_discrep-0.001, s=r'$D_\lambda$ for $\lambda \in \{0, 0.9\}$ (analytical)', ha='left', va='top')
plt.semilogx()
plt.ylim([-0.05, 0.2])
plt.ylabel('Q-function MSE')
plt.xlabel('Number of samples')

#%%
sns.lineplot(data=df.query('lambda_mode=="mc"'), x='n_samples', y='samp_discrep', hue='learning_rate', palette='colorblind')
plt.semilogx()
plt.hlines(lstd_discrep, df.n_samples.min(), df.n_samples.max(), linestyle='--', color='r')
plt.text(x=df.n_samples.median(), y=lstd_discrep-0.008, s=r'$D_\lambda$ for $\lambda \in \{0, 0.9\}$ (analytical)', ha='left')
plt.semilogx()
plt.ylabel(r'Estimated $\lambda$ discrepancy')
plt.xlabel('Number of samples')
