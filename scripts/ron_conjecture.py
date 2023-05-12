import numpy as np

from grl.environment import load_spec


spec_name = 'tmaze_5_two_thirds_up'
lamb = 0

spec = load_spec(spec_name)
Tsas = np.swapaxes(spec['T'], 0, 1)
Rsas = np.swapaxes(spec['R'], 0, 1)
phi = spec['phi']
gamma = spec['gamma']
s0 = spec['p0']

n = Tsas.shape[0]

pi = spec['Pi_phi'][0]

pis = phi @ pi
P = np.einsum('ik,ikj->ij',pis, Tsas)

Rsa = np.einsum('ijl,ijl->ij',Tsas,Rsas)
Rpisa = pis * Rsa
r = np.einsum('ij->i',Rpisa)

I = np.eye(n)

# TODO: this is functional_get_occupancy
mu = np.linalg.inv(I-gamma*P.T) @ s0
mu = mu/np.sum(mu)

D_mu = np.diag(mu)
Pi_D_mu = phi @ np.linalg.inv(phi.T @ D_mu @ phi) @ phi.T @ D_mu

A = phi.T @ D_mu @ (I - gamma * P) @ np.linalg.inv(I - gamma * lamb * P) @ phi
b = phi.T @ D_mu @ np.linalg.inv(I - gamma * lamb * P) @ r
V_LSTD_lamb = phi @ np.linalg.inv(A) @ b


def bellman(v):
    return r + gamma * P @ v

v = V_LSTD_lamb
amax = 25
a = np.empty((amax, n))
for i in range(amax):
    ai = Pi_D_mu @ v - V_LSTD_lamb
    ai[np.abs(ai) < 1e-10] = 0
    a[i] = ai
    print("a_" + str(i) + ": " + str(ai.round(4)))
    v = bellman(v)
