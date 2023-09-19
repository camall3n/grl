import copy

from jax import nn
import numpy as np
from tqdm import tqdm

from grl import environment
from grl.mdp import POMDP, MDP
from grl.agent.td_lambda import TDLambdaQFunction
from grl.utils.replaymemory import ReplayMemory

#%% Define base decision process
spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
pomdp = POMDP(mdp, spec['phi'])
pi_base = spec['Pi_phi'][0] # abstract policy over base (non-memory) actions
n_episodes = 10000

#%% Run TD-lambda until convergence
q_td = TDLambdaQFunction(n_obs=pomdp.observation_space.n,
                         n_actions=pomdp.action_space.n,
                         lambda_=0,
                         gamma=pomdp.gamma,
                         learning_rate=0.001)
q_mc = TDLambdaQFunction(n_obs=pomdp.observation_space.n,
                         n_actions=pomdp.action_space.n,
                         lambda_=0.99,
                         gamma=pomdp.gamma,
                         learning_rate=0.001)
replay = ReplayMemory(capacity=1000000)

for i in tqdm(range(n_episodes)):
    ob, _ = pomdp.reset()
    action = np.random.choice(mdp.action_space.n, p=pi_base[ob])
    terminal = False
    while not terminal:
        next_ob, reward, terminal, _, info = pomdp.step(action)
        next_action = np.random.choice(mdp.action_space.n, p=pi_base[next_ob])

        experience = {
            'obs': ob,
            'action': action,
            'reward': reward,
            'terminal': terminal,
            'next_obs': next_ob,
            'next_action': next_action,
        }

        q_td.update(**experience)
        q_mc.update(**experience)
        replay.push(experience)

        ob = next_ob
        action = next_action

q_mc_orig = copy.deepcopy(q_mc)
q_td_orig = copy.deepcopy(q_td)

#%% Define memory decision process (binary memory function)
n_mem_states = 2
n_mem_obs = pomdp.observation_space.n * pomdp.action_space.n * n_mem_states
initial_mem = 0
# p_hold = 0.95
# p_toggle = 1 - p_hold
# pi_mem_template = np.expand_dims(np.array([
#     [p_hold, p_toggle],
#     [p_toggle, p_hold],
# ]), axis=(0, 1))
# mem_params = pi_mem_template * np.ones((pomdp.action_space.n, pomdp.observation_space.n, n_mem_states, n_mem_states))
# mem_params = np.log(mem_params + 1e-5)
# mem_params += 0.5 * np.random.normal(size=(pomdp.action_space.n, pomdp.observation_space.n, n_mem_states, n_mem_states))
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
mem_params = np.log(memory_16 + 1e-5)
# mem_params = np.sqrt(2) * np.random.normal(size=(pomdp.action_space.n, pomdp.observation_space.n, n_mem_states, n_mem_states)).round(2)
lr = 0.01

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
pg_mode = None
# pg_mode = 'selected_action'
# pg_mode = 'all_actions'

q_mc = copy.deepcopy(q_mc_orig)
q_td = copy.deepcopy(q_td_orig)
q_td.augment_with_memory(n_mem_states)
q_mc.augment_with_memory(n_mem_states)

pi_aug = np.stack((pi_base, np.ones_like(pi_base) / pomdp.action_space.n),
                  axis=1).reshape(-1, pomdp.action_space.n)

#%%
n_episodes = 10000
for i in tqdm(range(n_episodes)):
    ob_base, _ = pomdp.reset()
    s_mem = initial_mem
    ob_aug = augment_obs(ob_base, s_mem, n_mem_states)

    # a_base = np.random.choice(pomdp.action_space.n, p=pi_base[ob_base])
    a_base = np.random.choice(pomdp.action_space.n, p=pi_aug[ob_aug])
    a_mem = np.random.choice(n_mem_states, p=pi_mem(a_base, ob_base, s_mem))

    param_updates = np.zeros_like(mem_params)
    total_discrepancy = 0.0

    terminal = False
    timestep = 0
    while not terminal:
        next_ob_base, r_base, terminal, _, _ = pomdp.step(a_base)
        next_s_mem = step_mem(s_mem, a_mem)
        next_ob_aug = augment_obs(next_ob_base, next_s_mem, n_mem_states)

        next_a_base = np.random.choice(mdp.action_space.n, p=pi_base[next_ob_base])
        next_a_mem = np.random.choice(n_mem_states,
                                      p=pi_mem(next_a_base, next_ob_base, next_s_mem))

        # update Q functions
        q_td.update(ob_aug, a_base, r_base, terminal, next_ob_aug, next_a_base)
        q_mc.update(ob_aug, a_base, r_base, terminal, next_ob_aug, next_a_base)

        if pg_mode == 'selected_action':
            # compute sampled (squared) lambda discrepancy
            # (Q_MC([ob+m]', a') - Q_TD([ob+m]', a'))
            step_discr = (q_mc.q[a_base, ob_aug] - q_td.q[a_base, ob_aug])**2
            total_discrepancy += step_discr

            # update policy using memory gradient
            #   minimize E[ discr \grad log π(m'|o,a,m)]
            param_updates[a_base, ob_base, s_mem, next_s_mem] -= 1
            param_updates[a_base, ob_base, s_mem, 1 - next_s_mem] += 1

        elif pg_mode == 'all_actions':
            all_s_mem_actions = [0, 1]
            for each_s_mem in all_s_mem_actions:
                each_ob_aug = augment_obs(next_ob_base, each_s_mem, n_mem_states)
                each_discr = (q_mc.q[next_a_base, each_ob_aug] -
                              q_td.q[next_a_base, each_ob_aug])**2
                pr_each_s_mem = pi_mem(a_base, ob_base, s_mem)[each_s_mem]

                # update policy using memory gradient
                #   minimize E[ discr \grad log π(m'|o,a,m)]
                param_updates[a_base, ob_base, s_mem, each_s_mem] -= each_discr * pr_each_s_mem
                param_updates[a_base, ob_base, s_mem, 1 - each_s_mem] += each_discr * pr_each_s_mem

        # increment timestep
        timestep += 1
        ob_base, a_base = next_ob_base, next_a_base
        s_mem, a_mem = next_s_mem, next_a_mem
        ob_aug = next_ob_aug

    mem_params += lr * total_discrepancy * param_updates

#%%
q_mc_orig.q
q_td_orig.q
np.abs(q_mc_orig.q - q_td_orig.q)

q_mc.q.round(3)
q_td.q.round(3)
np.abs(q_mc.q - q_td.q).round(4)

pi_base

nn.softmax(mem_params, axis=-1).round(4)[2, 0]
nn.softmax(mem_params, axis=-1).round(4)[2, 1]
nn.softmax(mem_params, axis=-1).round(4)[2, 2]

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
