import numpy as np

from mdp import MDP, AbstractMDP, one_hot
from vi import vi
from mc import mc

#%% ----

T1 = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

T2 = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

R1 = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

R2 = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

mdp = MDP([T1, T2], [R1, R2], gamma=0.5)
p0 = np.zeros(mdp.n_states)
p0[0] = 1

phi = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
amdp = AbstractMDP(mdp, phi)

Pi_phi = [
    np.array([0, 0, 0, 0]),
    np.array([1, 0, 1, 0]),
]

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=20000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='first', n_steps=2000)
    print(v, pi)

print()

for pi in [np.array([1, 0, 0, 0])]:
    v, q, pi = mc(mdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='all', n_steps=2000)
    print(v, pi)

#%% ----

T = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0.5, 0, 0.5],
    [0, 0, 0, 0]
])

R = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

mdp = MDP([T], [R], gamma=0.5)
p0 = np.zeros(mdp.n_states)
p0[0] = 1

phi = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])
amdp = AbstractMDP(mdp, phi)

Pi_phi = [
    np.array([0, 0, 0, 0]),
]

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=20000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='first', n_steps=20000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(mdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='all', n_steps=20000)
    print(v, pi)

#%% ----

T = np.array([
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

R = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

mdp = MDP([T], [R], gamma=0.5)
p0 = np.zeros(mdp.n_states)
p0[0] = 1

phi = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])
amdp = AbstractMDP(mdp, phi)

Pi_phi = [
    np.array([0, 0, 0, 0]),
]

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=50000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='first', n_steps=50000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(mdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=20000)
    print(v, pi)

#%% ----

T = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0.5, 0, 0.5],
    [0, 0, 0, 0]
])

R = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

mdp = MDP([T], [R], gamma=0.5)
p0 = np.zeros(mdp.n_states)
p0[0] = 1

phi = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
amdp = AbstractMDP(mdp, phi)

Pi_phi = [
    np.array([0, 0, 0, 0]),
]

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=50000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='first', n_steps=50000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(mdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=20000)
    print(v, pi)

#%% ----
# Example 13

T_up = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])
T_down = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

R = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

mdp = MDP([T_up, T_down], [R, R], gamma=0.5)
p0 = np.zeros(mdp.n_states)
p0[0] = 0.75
p0[1] = 0.25

phi = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
amdp = AbstractMDP(mdp, phi)

Pi_phi = [
    np.array([0, 0, 0, 0]),
    np.array([1, 1, 0, 0]),
]

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=50000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='first', n_steps=50000)
    print(v, pi)

print()

for pi in Pi_phi:
    v, q, pi = mc(mdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=20000)
    print(v, pi)
