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
            'lstd_q': lstd_q0,
            'samp_q': samp_q0,
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
            'lstd_q': lstd_q1,
            'samp_q': samp_q1,
        })

df = pd.DataFrame(data)

#%%
mc = df[np.isclose(df["lambda"], 0.95)]
td = df[np.isclose(df["lambda"], 0)]
lstd_discrep = (np.array(mc.lstd_q.values[0] - td.lstd_q.values[0])**2).mean()

samp_discreps = []
for mc_row, td_row, mc_nsamp, td_nsamp in zip(mc.samp_q.values, td.samp_q.values, mc.n_samples.values, td.n_samples.values):
    assert mc_nsamp == td_nsamp
    discrep = (np.array(mc_row - td_row)**2).mean()
    samp_discreps.append({
        'n_samples': mc_nsamp,
        'samp_discrep': discrep,
    })
samp_discreps = pd.DataFrame(samp_discreps)


#%%
sns.lineplot(data=df, x='lambda', y='value_error', hue='n_samples', style='n_samples', palette='colorblind')
plt.ylabel('Q-function MSE')
plt.xlabel('Lambda')

#%%
sns.lineplot(data=df.query('n_samples==1e6'), x='lambda', y='value_error', hue='n_samples', palette='colorblind')
plt.ylabel('Q-function MSE')
plt.xlabel('Lambda')

#%%
sns.lineplot(data=df.query('n_samples==1e7'), x='lambda', y='value_error', hue='n_samples', palette='colorblind')
plt.ylabel('Q-function MSE')
plt.xlabel('Lambda')

#%%
sns.lineplot(data=df, x='n_samples', y='value_error', hue='lambda')
plt.hlines(lstd_discrep, df.n_samples.min(), df.n_samples.max(), linestyle='--', color='r')
plt.text(x=df.n_samples.min(), y=lstd_discrep+0.03, s=r'$D_\lambda$ for $\lambda \in \{0, 1\}$ (analytical)', ha='left')
plt.semilogx()
plt.ylabel('Q-function MSE')
plt.xlabel('Number of samples')

#%%
sns.lineplot(data=samp_discreps, x='n_samples', y='samp_discrep')
plt.semilogx()
plt.hlines(lstd_discrep, df.n_samples.min(), df.n_samples.max(), linestyle='--', color='r')
plt.text(x=df.n_samples.median(), y=lstd_discrep-0.005, s=r'$D_\lambda$ for $\lambda \in \{0, 1\}$ (analytical)', ha='center')
plt.semilogx()
plt.ylabel(r'Estimated $\lambda$ discrepancy')
plt.xlabel('Number of samples')
