from functools import partial
import glob
import os

from jax import jit
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from grl import environment
from grl.mdp import POMDP, MDP
from grl.memory import memory_cross_product
from grl.utils.math import greedify
from grl.utils.mdp import amdp_get_occupancy
from grl.utils.policy_eval import functional_solve_mdp

#%%
experiment_name = 'exp22-tmaze5'
env_name = 'tmaze_5_two_thirds_up'

def get_perf(pi_obs: jnp.ndarray, env: POMDP):
    pi_state = env.phi @ pi_obs
    state_v, state_q = functional_solve_mdp(pi_state, env)
    return jnp.dot(env.p0, state_v)

results = {}
for results_dir in tqdm(glob.glob(f'results/sample_based/{experiment_name}/{env_name}/*/')):
    seed = int(os.path.basename(results_dir.rstrip("/")))
    # results_dir = f'results/sample_based/{experiment_name}/{env_name}/{seed}/'
    try:
        memory = np.load(results_dir + 'memory.npy')
        policy = np.load(results_dir + 'policy.npy')
        td = np.load(results_dir + 'q_td.npy')
        mc = np.load(results_dir + 'q_mc.npy')
        policy.round(3)
        memory.round(3)
        mc.round(2)
        td.round(2)
    except:
        print(f'File not found for seed {seed}')
        continue

    # study = load_study(experiment_name, env_name, seed)
    # params = [
    #     study.best_trial.params[key] for key in sorted(study.best_trial.params.keys(), key=int)
    # ]
    spec = environment.load_spec(env_name, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = POMDP(mdp, spec['phi'])
    mem_logits = jnp.log(memory + 1e-20)
    amdp_mem = memory_cross_product(mem_logits, env)

    def expected_lambda_discrep(amdp_mem, mem_logits, policy, td, mc):
        c_s = amdp_get_occupancy(greedify(policy), amdp_mem)
        c_o = (c_s @ amdp_mem.phi)
        p_o = c_o / c_o.sum()
        p_oa = (policy * p_o[:, None]).T
        return (abs(td - mc) * p_oa).sum()

    expected_lambda_discrep(amdp_mem, mem_logits, policy, td, mc)

    performance = get_perf(greedify(policy), amdp_mem)
    results[seed] = performance

for seed, performance in sorted(results.items(), key=lambda x: x[-1]):
    print(f'seed {seed}:', performance)
