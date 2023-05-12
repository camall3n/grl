import numpy as np

from grl.utils.math import one_hot

f = np.array([5, -2, 3, 4])
obs_aug = np.array([1, 2, 3, 2, 2, 1])

#%% ----- empirical mean -----
observed_f = f[obs_aug]
assert np.allclose(observed_f, np.array([-2,  3,  4,  3,  3, -2]))

empirical_mean_f = observed_f.mean()
assert np.allclose(empirical_mean_f, 1.5)


#%% ----- importance sampling (method 1) -----

# n_obs_aug = self.n_obs * self.n_mem_states
n_obs_aug = 4

obs_counts = one_hot(obs_aug, n_obs_aug).sum(axis=0)
assert np.allclose(obs_counts, np.array([0, 2, 3, 1]))

obs_mask = (obs_counts > 0)
alt_importance_weighted_mean_f = (f * obs_mask).sum() / obs_mask.sum()
assert np.isclose(alt_importance_weighted_mean_f, f[1:].mean())


#%% ----- importance sampling (method 2) -----

# n_obs_aug = self.n_obs * self.n_mem_states
n_obs_aug = 4

obs_counts = one_hot(obs_aug, n_obs_aug).sum(axis=0)
assert np.allclose(obs_counts, np.array([0, 2, 3, 1]))

obs_freq = obs_counts / obs_counts.sum()
assert np.allclose(obs_freq, np.array([0, 1/3, 1/2, 1/6]))

n_unique_obs = (obs_counts > 0).sum()
assert n_unique_obs == 3

importance_weights = (1 / n_unique_obs) / (obs_freq + 1e-12) * (obs_counts > 0)
assert np.allclose(importance_weights, np.array([0, 1, 2/3, 2]))

weighted_discreps = observed_f * importance_weights[obs_aug]
importance_weighted_mean_f = weighted_discreps.mean()
assert np.allclose(importance_weighted_mean_f, f[1:].mean())
