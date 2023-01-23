import argparse
import copy
import os

import numpy as np
from tqdm import tqdm
import optuna

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='tmaze_5_two_thirds_up')
    parser.add_argument('--study_name', type=str, default='exp03-mi')
    parser.add_argument('--n_memory_trials', type=int, default=1000)
    parser.add_argument('--n_memory_iterations', type=int, default=20)
    parser.add_argument('--n_policy_iterations', type=int, default=100)
    parser.add_argument('--n_episodes_per_policy', type=int, default=20000)
    return parser.parse_args()

global args
args = parse_args()

def converge_value_functions(agent, env):
    for i in range(args.n_episodes_per_policy):
        agent.reset()
        obs, _ = env.reset()
        action = agent.act(obs)
        terminal = False
        while not terminal:
            next_obs, reward, terminal, _, info = env.step(action)
            next_action = agent.act(next_obs)

            experience = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'next_obs': next_obs,
                'next_action': next_action,
            }

            agent.update_critic(experience)
            agent.store(experience)

            obs = next_obs
            action = next_action

def optimize_policy(agent, env):
    for i in tqdm(range(args.n_policy_iterations)):
        print(f'Policy iteration: {i}')
        print(agent.cached_policy_fn)
        converge_value_functions(agent, env)
        did_change = agent.update_actor()
        if not did_change:
            break

def cpu_count():
    # os.cpu_count()
    #     returns number of cores on machine
    # os.sched_getaffinity(pid)
    #     returns set of cores on which process is allowed to run
    #     if pid=0, results are for current process
    #
    # if os.sched_getaffinity doesn't exist, just return cpu_count and hope for the best
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

def get_n_workers(n_tasks):
    workers_available = max(1, cpu_count())
    workers_needed = n_tasks
    n_workers = min(workers_needed, workers_available)
    return n_workers

def main():
    parse_args()
    spec = environment.load_spec(args.env, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    agent = ActorCritic(n_obs=env.n_obs,
                        n_actions=env.n_actions,
                        gamma=env.gamma,
                        n_mem_entries=0,
                        replay_buffer_size=int(4e6))
    agent.reset_policy()

    for n_mem_iterations in range(args.n_memory_iterations):
        print(f"Memory iteration {n_mem_iterations}")
        optimize_policy(agent, env)

        if agent.n_mem_entries == 0:
            agent.add_memory()

        agent.optimize_memory(
            f'{args.env}/{args.study_name}',
            n_jobs=get_n_workers(args.n_memory_trials),
            n_trials=args.n_memory_trials,
        )

        print('Memory:')
        print(agent.cached_memory_fn.round(3))
        print()
        print('Policy:')
        print(agent.cached_policy_fn)


    agent.reset_policy()
    optimize_policy(agent, env)

    print('Final memory:')
    print(agent.cached_memory_fn.round(3))
    print()
    print('Final policy:')
    print(agent.cached_policy_fn)

if __name__ == '__main__':
    main()
