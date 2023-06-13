import argparse
import os
from pathlib import Path
from typing import Union
from jax.config import config

import numpy as np
from tqdm import tqdm

from grl import environment
from grl.agent.actorcritic import ActorCritic
from grl.mdp import POMDP, MDP
from grl.memory import memory_cross_product
from grl.utils.file_system import numpyify_and_save
from grl.utils.policy_eval import lstdq_lambda

cast_as_int = lambda x: int(float(x))

def log_info(agent: ActorCritic, amdp: POMDP) -> dict:
    pi = agent.policy_probs
    info = {'lambda_0': agent.lambda_0, 'lambda_1': agent.lambda_1, 'policy_probs': pi.copy()}

    sample_based_info = {'q0': agent.q_td.q, 'q1': agent.q_mc.q}

    if agent.n_mem_entries > 0:
        amdp = memory_cross_product(agent.memory_logits, amdp)
        info['memory_probs'] = agent.memory_probs.copy()

        # save previous memory, reset and step through all obs and actions
        all_obs, all_actions = agent.replay.retrieve(fields=['obs', 'action'])
        agent.reset_memory_state()

        memories = []
        for obs, action in zip(all_obs, all_actions):
            memories.append(agent.memory)
            agent.step_memory(obs, action)

        sample_based_info['discrepancy_loss'] = agent.compute_discrepancy_loss(
            all_obs, all_actions, memories)

    lstd_v0, lstd_q0, _ = lstdq_lambda(pi, amdp, lambda_=agent.lambda_0)
    lstd_v1, lstd_q1, _ = lstdq_lambda(pi, amdp, lambda_=agent.lambda_1)
    analytical_info = {'q0': lstd_q0, 'q1': lstd_q1}

    info['sample_based'] = sample_based_info
    info['analytical'] = analytical_info

    return info

def log_and_save_info(agent: ActorCritic, amdp: POMDP, save_path: Union[Path, str]):
    info = log_info(agent, amdp)
    numpyify_and_save(save_path, info)

def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='tmaze_5_two_thirds_up')
    parser.add_argument('--study_name', type=str, default='test')
    parser.add_argument('--max_jobs', type=int, default=None)
    parser.add_argument('--load_policy', action='store_true')
    parser.add_argument('--policy_junction_up_prob', type=float, default=None)
    parser.add_argument('--policy_epsilon', type=float, default=None)
    parser.add_argument('--lambda0', type=float, default=0.0)
    parser.add_argument('--lambda1', type=float, default=0.99999)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--trial_id', default=1)
    parser.add_argument('--n_memory_trials', type=int, default=500)
    parser.add_argument('--n_memory_iterations', type=int, default=2)
    parser.add_argument('--n_policy_iterations', type=int, default=10)
    parser.add_argument('--n_samples_per_policy', type=int, default=2e6)
    parser.add_argument('--min_mem_opt_replay_size', type=int, default=2e6,
        help="Minimum number of experiences in replay buffer for memory optimization")
    # parser.add_argument('--use_min_replay_num_samples', action='store_true')
    parser.add_argument('--replay_buffer_size', type=cast_as_int, default=4e6)
    parser.add_argument('--mellowmax_beta', type=float, default=50.)
    parser.add_argument('--use_existing_study', action='store_true')
    parser.add_argument('--discrep_loss', type=str, default='mse', choices=['abs', 'mse'])
    parser.add_argument('--disable_importance_sampling', action='store_true')
    parser.add_argument('--analytical_mem_eval', action='store_true')
    parser.add_argument('-f', help='fool ipython')
    # yapf: enable
    return parser.parse_args()

def converge_value_functions(agent: ActorCritic,
                             env,
                             n_samples,
                             mode='td',
                             reset_before_converging=False,
                             update_policy=False):
    if reset_before_converging:
        agent.reset_value_functions()

    pbar = tqdm(total=n_samples)
    total_samples = 0
    while total_samples < n_samples:
        agent.reset_memory_state()
        obs, _ = env.reset()
        action = agent.act(obs)
        terminal = False
        eps_length = 0
        while not terminal:
            agent.step_memory(obs, action)
            next_obs, reward, terminal, _, _ = env.step(action)
            eps_length += 1
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

        if update_policy:
            agent.update_actor(mode=mode, argmax_type='hardmax')
        pbar.update(eps_length)
        total_samples += eps_length

    # td_v0s = []
    # mc_v0s = []
    # for i in range(100):
    #     obs, _ = env.reset()
    #     td_v0s.append(np.dot(agent.q_td.q[:, obs], agent.policy_probs[obs, :]))
    #     mc_v0s.append(np.dot(agent.q_mc.q[:, obs], agent.policy_probs[obs, :]))
    # td_v0 = np.mean(td_v0s)
    # mc_v0 = np.mean(mc_v0s)
    # print(f"td_v0: {td_v0}")
    # print(f"mc_v0: {mc_v0}")
    np.save(agent.study_dir + '/q_mc.npy', agent.q_mc.q)
    np.save(agent.study_dir + '/q_td.npy', agent.q_td.q)

def optimize_policy(agent: ActorCritic, env, mode='td', args=parse_args()):
    for i in tqdm(range(args.n_policy_iterations)):
        print(f'Policy iteration: {i}')
        print('Policy:\n', agent.policy_probs.round(4))
        print('Q(TD):\n', agent.q_td.q.T.round(5))
        print('Q(MC):\n', agent.q_mc.q.T.round(5))
        converge_value_functions(agent,
                                 env,
                                 args.n_samples_per_policy,
                                 mode=mode,
                                 reset_before_converging=True,
                                 update_policy=True)
        np.save(agent.study_dir + '/policy.npy', agent.policy_probs)

        # did_change = agent.update_actor(mode=mode, argmax_type='mellowmax')
        # if not did_change:
        #     break

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

def get_n_workers(n_tasks, args):
    workers_available = max(1, cpu_count())
    workers_needed = n_tasks
    n_workers = min(workers_needed, workers_available)
    if args.max_jobs is not None:
        n_workers = min(n_workers, args.max_jobs)
    return n_workers

def main():
    config.update('jax_platform_name', 'cpu')
    np.set_printoptions(suppress=True)

    args = parse_args()
    # if args.use_min_replay_num_samples:
    #     args.min_mem_opt_replay_size = args.replay_buffer_size
    spec = environment.load_spec(args.env, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = POMDP(mdp, spec['phi'])

    agent = ActorCritic(
        n_obs=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=env.gamma,
        n_mem_entries=0,
        policy_epsilon=args.policy_epsilon,
        replay_buffer_size=args.replay_buffer_size,
        mellowmax_beta=args.mellowmax_beta,
        study_name=f'{args.study_name}/{args.env}/{args.trial_id}',
        use_existing_study=args.use_existing_study,
        discrep_loss=args.discrep_loss,
        disable_importance_sampling=args.disable_importance_sampling,
        override_mem_eval_with_analytical_env=env if args.analytical_mem_eval else None,
    )
    info = {'args': args.__dict__}

    if args.load_policy:
        policy = spec['Pi_phi'][0]
        if args.policy_junction_up_prob is not None:
            assert args.env == 'tmaze_5_two_thirds_up'
            policy[3][0] = args.policy_junction_up_prob
            policy[3][1] = 1 - args.policy_junction_up_prob
        if args.policy_epsilon is not None:
            uniform = np.ones_like(policy, dtype=float) / policy.shape[-1]
            policy = (1 - args.policy_epsilon) * policy + args.policy_epsilon * uniform
        agent.set_policy(policy, logits=False) # policy over non-memory observations
    else:
        agent.reset_policy()

    if not args.load_policy:
        optimize_policy(agent, env, args=args)

    agent.add_memory()
    agent.reset_memory()

    while len(agent.replay) < args.min_mem_opt_replay_size:
        converge_value_functions(agent, env, args.n_samples_per_policy)

    initial_mem_info_path = agent.study_dir + '/initial_mem_info.pkl'
    log_and_save_info(agent, env, initial_mem_info_path)

    required_params = list(agent.memory_probs[:, :, :, :-1].flatten())
    assert np.allclose(agent.memory_probs, agent.fill_in_params(required_params))

    for n_mem_iterations in range(args.n_memory_iterations):
        print(f"Memory iteration {n_mem_iterations}")

        study = agent.optimize_memory(
            n_jobs=get_n_workers(args.n_memory_trials, args),
            n_trials=args.n_memory_trials,
        )
        np.save(agent.study_dir + '/memory.npy', agent.memory_probs)

        print('Memory:')
        print(agent.memory_probs.round(3))
        print()
        print('Policy:')
        print(agent.policy_probs)

        if not args.load_policy:
            agent.reset_policy()
            optimize_policy(agent, env, args)

        print('Memory:')
        print(agent.memory_probs.round(3))
        print()
        print('Policy:')
        print(agent.policy_probs)

        while len(agent.replay) < args.min_mem_opt_replay_size:
            converge_value_functions(agent, env, args.n_samples_per_policy)

        mem_iter_info_path = agent.study_dir + f'/mem_iter_{n_mem_iterations}_info.pkl'
        log_and_save_info(agent, env, mem_iter_info_path)

    print('Final memory:')
    print(agent.memory_probs.round(3))
    print()
    print('Final policy:')
    print(agent.policy_probs)
    np.save(agent.study_dir + '/memory.npy', agent.memory_probs)
    np.save(agent.study_dir + '/policy.npy', agent.policy_probs)
    np.save(agent.study_dir + '/q_mc.npy', agent.q_mc.q)
    np.save(agent.study_dir + '/q_td.npy', agent.q_td.q)
    info.update({
        'final_params': agent.memory_logits,
        'final_discrep': study.best_value,
        'policy_up_prob': args.policy_junction_up_prob,
        'final_memory_value_function': {
            '0': agent.q_td.q.copy(),
            '1': agent.q_mc.q.copy()
        },
        'policy_epsilon': args.policy_epsilon,
    })
    np.save(agent.study_dir + '/info.npy', info)

if __name__ == '__main__':
    main()
