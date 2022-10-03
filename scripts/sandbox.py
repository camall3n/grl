import numpy as np

from grl import environment
from grl.mdp import AbstractMDP, MDP

from grl.agents.td_lambda import TDLambdaQFunction
#%% Define base decision process
corridor_length = 5
spec = environment.load_spec('tmaze_5_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
amdp = AbstractMDP(mdp, spec['phi'])
pi_base = spec['Pi_phi'][0] # abstract policy over base (non-memory) actions
n_episodes = 1000

#%% Run TD-lambda until convergence
q_td = TDLambdaQFunction(amdp.n_obs,
                         amdp.n_actions,
                         lambda_=0,
                         gamma=amdp.gamma,
                         learning_rate=0.1)
q_mc = TDLambdaQFunction(amdp.n_obs,
                         amdp.n_actions,
                         lambda_=0.99,
                         gamma=amdp.gamma,
                         learning_rate=0.1)

for i in range(n_episodes):
    s = np.random.choice(mdp.n_states, p=mdp.p0)
    ob = mdp.observe(s)
    a = np.random.choice(mdp.n_actions, p=pi_base[ob])
    done = False
    while not done:
        next_s, r, done = mdp.step(s, a, mdp.gamma)
        next_ob = mdp.observe(next_s)
        next_a = np.random.choice(mdp.n_actions, p=pi_base[next_ob])

        q_td.update(ob, a, r, done, next_ob, next_a)
        q_mc.update(ob, a, r, done, next_ob, next_a)

        s = next_s
        ob = next_ob
        a = next_a

#%% Define memory decision process (binary memory function)
n_mem_obs = amdp.n_obs * amdp.n_actions * 2
n_mem_states = 2
n_mem_actions = 2 # {hold, toggle}
initial_mem = 0

def get_ob_mem(ob_base, a_base, s_mem):
    ob_mem = (a_base * amdp.n_obs + ob_base) * n_mem_states + s_mem
    if ob_mem >= n_mem_obs:
        raise RuntimeError(f'Bad state detected for memory decision process.\n'
                           f'  ob_base: {ob_base}\n'
                           f'  a_base:  {a_base}\n'
                           f'  s_mem:   {s_mem}\n'
                           f'  ob_mem:  {ob_mem}  (should be < {n_mem_obs})')
    return ob_mem

for i in range(n_episodes):
    s_mem = initial_mem
    s_base = np.random.choice(amdp.n_states, p=mdp.p0)
    ob_base = mdp.observe(s_base)
    a_base = np.random.choice(amdp.n_actions, p=pi_base[ob_base])

    ob_mem = get_ob_mem(ob_base, a_base, s_mem)

    terminal = False
    while not terminal:
        next_s_base, r_base, terminal = amdp.step(s_base, a_base)
        next_ob_base = amdp.observe(next_s)
        next_a_base = np.random.choice(mdp.n_actions, p=pi_base[next_ob])

        q_td.update(ob_base, a_base, r_base, terminal, next_ob_base, next_a_base)
        q_mc.update(ob_base, a_base, r_base, terminal, next_ob_base, next_a_base)

        s_base = next_s_base
        ob_base = next_ob_base
        a_base = next_a_base
