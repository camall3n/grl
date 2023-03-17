import numpy as np
import matplotlib.pyplot as plt

from grl.environment import load_spec

spec_name = 'tmaze_5_two_thirds_up'
lamb = 0

# Actions are always first! So a state-action pair is (a, s)

spec = load_spec(spec_name)
T_ass = spec['T']
R_ass = spec['R']
phi = spec['phi']
gamma = spec['gamma']
s0 = spec['p0']

n = T_ass.shape[1]
a = T_ass.shape[0]
o = phi.shape[1]
na = n*a
oa = o*a

pi = spec['Pi_phi'][0]
pi_sa = phi @ pi
plt.imshow(pi_sa)

as_0 = (s0[:,None] * pi_sa).T.reshape(na)

P_asas = np.einsum('ijk,kl->ijlk', T_ass, pi_sa)
P_as_as = P_asas.reshape((na, na))
plt.imshow(P_as_as)
plt.vlines(-0.5 + n * np.arange(a), -0.5, na - 0.5)
plt.hlines(-0.5 + n * np.arange(a), -0.5, na - 0.5)

R_as = np.einsum('ijk,ijk->ij', T_ass, R_ass).reshape((na,))

I = np.eye(na)

# TODO: this is functional_get_occupancy
mu = np.linalg.inv(I - gamma * P_as_as.T) @ as_0
mu = mu / np.sum(mu)
D_mu = np.diag(mu)

plt.imshow(D_mu)
plt.vlines(-0.5 + n * np.arange(a), -0.5, na - 0.5)
plt.hlines(-0.5 + n * np.arange(a), -0.5, na - 0.5)

phi_as_ao = np.kron(np.eye(a), phi)
plt.imshow(phi_as_ao)
plt.xlim([-0.5, oa - 0.5])
plt.ylim([na - 0.5, -0.5])
plt.vlines(-0.5 + o * np.arange(a), -0.5, na - 0.5)
plt.hlines(-0.5 + n * np.arange(a), -0.5, oa - 0.5)

D_mu_ao_ao = phi_as_ao.T @ D_mu @ phi_as_ao
D_eps_ao = 1e-10 * np.eye(oa)
D_mu_ao_ao[D_mu_ao_ao > 0]
plt.imshow(D_mu_ao_ao)
plt.vlines(-0.5 + o * np.arange(a), -0.5, oa - 0.5)
plt.hlines(-0.5 + o * np.arange(a), -0.5, oa - 0.5)

Pi_D_mu = phi_as_ao @ np.linalg.inv(D_mu_ao_ao + D_eps_ao) @ phi_as_ao.T @ D_mu
plt.imshow(Pi_D_mu)
plt.vlines(-0.5 + n * np.arange(a), -0.5, na - 0.5)
plt.hlines(-0.5 + n * np.arange(a), -0.5, na - 0.5)

A = phi_as_ao.T @ D_mu @ (I - gamma * P_as_as) @ np.linalg.inv(I - gamma * lamb * P_as_as) @ phi_as_ao
b = phi_as_ao.T @ D_mu @ np.linalg.inv(I - gamma * lamb * P_as_as) @ R_as
Q_LSTD_lamb_an = (phi_as_ao @ np.linalg.inv(A + D_eps_oa) @ b).reshape((a, n))
V_hat = np.einsum('ij,ji->j', Q_LSTD_lamb_an, pi_sa)


V_LSTD_lamb = np.array([
    1.03630995, 1.03630995, 1.1514555 , 1.1514555 , 1.1514555 ,
    1.1514555 , 1.1514555 , 1.1514555 , 1.1514555 , 1.1514555 ,
    1.1514555 , 1.1514555 , 1.95      , 1.95      , 0.
])

V_hat - V_LSTD_lamb


# def bellman(v):
#     return r + gamma * P @ v
#
# v = V_LSTD_lamb
# amax = 25
# a = np.empty((amax, n))
# for i in range(amax):
#     ai = Pi_D_mu @ v - V_LSTD_lamb
#     ai[np.abs(ai) < 1e-10] = 0
#     a[i] = ai
#     print("a_" + str(i) + ": " + str(ai.round(4)))
#     v = bellman(v)
