import copy
from tqdm import tqdm

import numpy as np

from grl import environment
from grl.mdp import AbstractMDP, MDP

from jax import nn

from grl.agents.td_lambda import TDLambdaQFunction

#%% Define base decision process
spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
amdp = AbstractMDP(mdp, spec['phi'])
pi_base = spec['Pi_phi'][0] # abstract policy over base (non-memory) actions
n_episodes = 10000

#%% Run TD-lambda until convergence
q_td = TDLambdaQFunction(n_observations=amdp.n_obs,
                         n_actions=amdp.n_actions,
                         lambda_=0,
                         gamma=amdp.gamma,
                         learning_rate=0.001)
q_mc = TDLambdaQFunction(n_observations=amdp.n_obs,
                         n_actions=amdp.n_actions,
                         lambda_=0.99,
                         gamma=amdp.gamma,
                         learning_rate=0.001)

for i in range(n_episodes):
    s = np.random.choice(mdp.n_states, p=mdp.p0)
    ob = amdp.observe(s)
    a = np.random.choice(mdp.n_actions, p=pi_base[ob])
    s, ob, a
    terminal = False
    while not terminal:
        s, ob, a
        next_s, r, terminal = mdp.step(s, a, mdp.gamma)
        r, terminal
        next_ob = amdp.observe(next_s)
        next_a = np.random.choice(mdp.n_actions, p=pi_base[next_ob])
        next_s, next_ob, next_a

        q_td.update(ob, a, r, terminal, next_ob, next_a)
        q_mc.update(ob, a, r, terminal, next_ob, next_a)

        s = next_s
        ob = next_ob
        a = next_a

# TODO: archive non-augmented q functions to use as baselines?
q_mc_orig = copy.deepcopy(q_mc)
q_td_orig = copy.deepcopy(q_td)

#%% Define memory decision process (binary memory function)
n_mem_states = 2
n_mem_obs = amdp.n_obs * amdp.n_actions * n_mem_states
initial_mem = 0
# p_hold = 0.95
# p_toggle = 1 - p_hold
# pi_mem_template = np.expand_dims(np.array([
#     [p_hold, p_toggle],
#     [p_toggle, p_hold],
# ]), axis=(0, 1))
# mem_params = pi_mem_template * np.ones((amdp.n_actions, amdp.n_obs, n_mem_states, n_mem_states))
# mem_params = np.log(mem_params+1e-5)
# mem_params += 0.5 * np.random.normal(size=(amdp.n_actions, amdp.n_obs, n_mem_states, n_mem_states))
# Optimal memory for t-maze
mem_16 = np.array([
    [ # we see the goal as UP
        # Pr(m'| m, o)
        # m0', m1'
        [1., 0], # m0
        [1, 0], # m1
    ],
    [ # we see the goal as DOWN
        [0, 1],
        [0, 1],
    ],
    [ # corridor
        [1, 0],
        [0, 1],
    ],
    [ # junction
        [1, 0],
        [0, 1],
    ],
    [ # terminal
        [1, 0],
        [0, 1],
    ],
])
memory_16 = np.array([mem_16, mem_16, mem_16, mem_16]) # up, down, right, left
mem_params = np.log(memory_16+1e-5)
mem_params = 0.1 * np.random.normal(size=(amdp.n_actions, amdp.n_obs, n_mem_states, n_mem_states))
lr = 0.1

def pi_mem(a_base, ob_base, s_mem):
    logits = mem_params[a_base, ob_base, s_mem]
    # return logits
    return nn.softmax(logits, axis=-1)

def step_mem(s_mem, a_mem):
    # next_s_mem = ~s_mem if (a_mem == 1) else s_mem
    # next_s_mem = (1 - a_mem) * s_mem + a_mem * (1 - s_mem)  # hold/toggle interpretation
    next_s_mem = a_mem # set/reset interpretation
    return next_s_mem

def augment_obs(ob_base, s_mem, n_mem_states):
    # augment last dim with mem states
    ob_augmented = n_mem_states * ob_base + s_mem
    return ob_augmented

#%%
q_mc = copy.deepcopy(q_mc_orig)
q_td = copy.deepcopy(q_td_orig)
q_td.augment_with_memory(n_mem_states)
q_mc.augment_with_memory(n_mem_states)

#%%
n_episodes = 10000
for i in tqdm(range(n_episodes)):
    s_base = np.random.choice(amdp.n_states, p=mdp.p0)
    ob_base = amdp.observe(s_base)
    a_base = np.random.choice(amdp.n_actions, p=pi_base[ob_base])

    s_mem = initial_mem
    a_mem = np.random.choice(n_mem_states, p=pi_mem(a_base, ob_base, s_mem))

    ob_aug = augment_obs(ob_base, s_mem, n_mem_states)

    param_updates = np.zeros_like(mem_params)

    terminal = False
    timestep = 0
    while not terminal:
        next_s_base, r_base, terminal = amdp.step(s_base, a_base)
        next_ob_base = amdp.observe(next_s_base)
        next_a_base = np.random.choice(mdp.n_actions, p=pi_base[next_ob_base])

        next_s_mem = step_mem(s_mem, a_mem)
        next_a_mem = np.random.choice(n_mem_states,
                                      p=pi_mem(next_a_base, next_ob_base, next_s_mem))

        next_ob_aug = augment_obs(next_ob_base, next_s_mem, n_mem_states)

        # update Q functions
        q_td.update(ob_aug, a_base, r_base, terminal, next_ob_aug, next_a_base)
        q_mc.update(ob_aug, a_base, r_base, terminal, next_ob_aug, next_a_base)

        # TODO: use all-actions method over next_s_mem?

        # compute sampled (squared) lambda discrepancy
        # (R + \gamma G_{t+1}) - (R + \gamma Q_TD([ob+m]', a'))
        # (Q_MC([ob+m]', a') - Q_TD([ob+m]', a'))
        step_discr = (q_mc.q[next_a_base, next_ob_aug] - q_td.q[next_a_base, next_ob_aug])**2

        # update policy using memory gradient
        #   minimize E[ discr \grad log π(m'|o,a,m)]
        # param_updates[a_base, ob_base, s_mem, next_s_mem] -= step_discr
        # param_updates[a_base, ob_base, s_mem, 1 - next_s_mem] += step_discr

        # increment timestep
        timestep += 1
        s_base, ob_base, a_base = next_s_base, next_ob_base, next_a_base
        s_mem, a_mem = next_s_mem, next_a_mem
        ob_aug = next_ob_aug

    # mem_params += lr * param_updates
#%%
q_mc_orig.q

q_td_orig.q

(q_mc_orig.q - q_td_orig.q)**2

q_mc.q.round(4)
q_td.q.round(4)

((q_mc.q - q_td.q)**2).round(4)

pi_base

nn.softmax(mem_params, axis=-1).round(2)[2, 0]
nn.softmax(mem_params, axis=-1).round(2)[2, 1]
nn.softmax(mem_params, axis=-1).round(2)[2, 2]

#%%

# Cam Notes
#
# Good:
# - Sample-based value functions appear accurate.
# - Memory augmentation works.
# - Sample-based augmented value functions are also correct.
# - No parameter updates for expert memory function (which is expected behavior).
#
# Bad:
# - Memory gradient doesn't converge to expert memory function
