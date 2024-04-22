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

reward_range_dict = {
    'cheese.95': (10.0, 0),
    'tiger-alt-start': (10.0, -100.0),
    'network': (80.0, -40.0),
    'tmaze_5_two_thirds_up': (4.0, -0.1),
    'example_7': (1.0, 0.0),
    '4x3.95': (1.0, -1.0),
    'shuttle.95': (10.0, -3.0),
    'paint.95': (1.0, -1.0),
    'bridge-repair': (4018, 0),
    'hallway': (1.0, 0),
}

def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='tmaze_5_two_thirds_up')
    parser.add_argument('--study_name', type=str, default='test')
    parser.add_argument('--max_jobs', type=int, default=None)
    parser.add_argument('--load_policy', action='store_true')
    parser.add_argument('--policy_optim_alg', type=str, default='policy_iter',
                        choices=['policy_iter', 'policy_grad'])
    parser.add_argument('--policy_optim_lr', type=float, default=0.01)
    parser.add_argument('--init_policy_randomly', action='store_true')
    parser.add_argument('--n_random_policies', type=int, default=400)
    parser.add_argument('--policy_junction_up_prob', type=float, default=None)
    parser.add_argument('--policy_epsilon', type=float, default=0.1)
    parser.add_argument('--lambda0', type=float, default=0.0)
    parser.add_argument('--lambda1', type=float, default=0.99999)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--trial_id', default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--normalize_reward_range', action='store_true')
    parser.add_argument('--mem_optimizer', type=str, default='queue',
                        choices=['queue', 'annealing', 'optuna'])
    parser.add_argument('--mem_optim_objective', type=str, default='ld', choices=['ld', 'td'])
    parser.add_argument('--enable_priority_queue', action='store_true')
    parser.add_argument('--annealing_should_sample_hyperparams', action='store_true')
    parser.add_argument('--annealing_tmax', type=float, default=3.16e-3)
    parser.add_argument('--annealing_tmin', type=float, default=1e-7)
    parser.add_argument('--annealing_progress_fraction_at_tmin', type=float, default=0.3)
    parser.add_argument('--n_annealing_repeats', type=int, default=10)
    parser.add_argument('--n_memory_trials', type=int, default=500)
    parser.add_argument('--n_memory_iterations', type=int, default=2)
    parser.add_argument('--n_memory_states', type=int, default=2)
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
                             update_policy=False,
                             reward_scale=1.0):
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
            effective_reward = reward * reward_scale
            eps_length += 1
            next_action = agent.act(next_obs)

            experience = {
                'obs': obs,
                'action': action,
                'reward': effective_reward,
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
    return total_samples

def optimize_policy(agent: ActorCritic,
                    env,
                    n_policy_iterations,
                    n_samples_per_policy,
                    mode='td',
                    reward_scale=1.0):
    policy_history = []
    policy_history.append((0, agent.policy_probs))
    for i in tqdm(range(n_policy_iterations)):
        print(f'Policy iteration: {i}')
        print('Policy:\n', agent.policy_probs.round(4))
        print('Q(TD):\n', agent.q_td.q.T.round(5))
        print('Q(MC):\n', agent.q_mc.q.T.round(5))
        n_samples = converge_value_functions(agent,
                                             env,
                                             n_samples=n_samples_per_policy,
                                             mode=mode,
                                             reset_before_converging=True,
                                             update_policy=False)
        np.save(agent.study_dir + '/policy.npy', agent.policy_probs)

        did_change = agent.update_actor(mode=mode, argmax_type='hardmax')
        policy_history.append((n_samples, agent.policy_probs))
        # if not did_change:
        #     break
    return policy_history

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

def log_info(agent: ActorCritic, pomdp: POMDP) -> dict:
    pi = agent.policy_probs
    info = {'lambda_0': agent.lambda_0, 'lambda_1': agent.lambda_1, 'policy_probs': pi.copy()}

    sample_based_info = {'q0': agent.q_td.q, 'q1': agent.q_mc.q}

    if agent.n_mem_entries > 0:
        pomdp = memory_cross_product(agent.memory_logits, pomdp)
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

    lstd_v0, lstd_q0, _ = lstdq_lambda(pi, pomdp, lambda_=agent.lambda_0)
    lstd_v1, lstd_q1, _ = lstdq_lambda(pi, pomdp, lambda_=agent.lambda_1)
    analytical_info = {'q0': lstd_q0, 'q1': lstd_q1}

    info['sample_based'] = sample_based_info
    info['analytical'] = analytical_info

    return info

def log_and_save_info(agent: ActorCritic, pomdp: POMDP, save_path: Union[Path, str]):
    info = log_info(agent, pomdp)
    numpyify_and_save(save_path, info)

def main():
    config.update('jax_platform_name', 'cpu')
    np.set_printoptions(suppress=True)

    args = parse_args()
    np.random.seed(args.seed)

    # if args.use_min_replay_num_samples:
    #     args.min_mem_opt_replay_size = args.replay_buffer_size
    spec = environment.load_spec(args.env, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = POMDP(mdp, spec['phi'])

    reward_scale = args.reward_scale
    if args.normalize_reward_range and args.env in reward_range_dict:
        reward_scale = 1 / (reward_range_dict[args.env][0] - reward_range_dict[args.env][1])

    agent = ActorCritic(
        n_obs=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=env.gamma,
        n_mem_entries=0,
        n_mem_values=args.n_memory_states,
        policy_epsilon=args.policy_epsilon,
        replay_buffer_size=args.replay_buffer_size,
        mem_optimizer=args.mem_optimizer,
        mem_optim_objective=args.mem_optim_objective,
        ignore_queue_priority=(not args.enable_priority_queue),
        annealing_should_sample_hyperparams=args.annealing_should_sample_hyperparams,
        annealing_tmax=args.annealing_tmax,
        annealing_tmin=args.annealing_tmin,
        annealing_progress_fraction_at_tmin=args.annealing_progress_fraction_at_tmin,
        n_annealing_repeats=args.n_annealing_repeats,
        prune_if_parent_suboptimal=False,
        mellowmax_beta=args.mellowmax_beta,
        study_name=f'{args.study_name}/{args.env}/{args.trial_id}',
        use_existing_study=args.use_existing_study,
        n_optuna_workers=get_n_workers(args.n_memory_trials, args),
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

    initial_policy_history = None
    if not args.load_policy:
        optimize_policy(
            agent,
            env,
            n_policy_iterations=args.n_policy_iterations,
            n_samples_per_policy=args.n_samples_per_policy,
            reward_scale=reward_scale,
        )

    agent.add_memory()
    agent.reset_memory()

    def maybe_fill_replay_buffer():
        n_remaining_samples = args.min_mem_opt_replay_size - len(agent.replay)
        if n_remaining_samples > 0:
            print("Adding more samples to replay buffer to meet minimum requirements")
            converge_value_functions(agent, env, n_remaining_samples, reward_scale=reward_scale)

    maybe_fill_replay_buffer()

    discrep_start = agent.evaluate_memory()
    initial_mem_info_path = agent.study_dir + '/initial_mem_info.pkl'
    log_and_save_info(agent, env, initial_mem_info_path)

    required_params = list(agent.memory_probs[:, :, :, :-1].flatten())
    assert np.allclose(agent.memory_probs, agent.fill_in_params(required_params))

    for n_mem_iterations in range(args.n_memory_iterations):
        maybe_fill_replay_buffer()

        print(f"Memory iteration {n_mem_iterations}")

        t_max = 1e-1 * discrep_start / 0.225
        t_min = 1e-4 * discrep_start / 0.225
        optim_results = agent.optimize_memory(n_trials=args.n_memory_trials)
        np.save(agent.study_dir + '/memory.npy', agent.memory_probs)

        print('Memory:')
        print(agent.memory_probs.round(3))
        print()
        print('Policy:')
        print(agent.policy_probs)

        mem_opt_policy_history = None
        if not args.load_policy:
            agent.reset_policy()
            mem_opt_policy_history = optimize_policy(
                agent,
                env,
                args.n_policy_iterations,
                args.n_samples_per_policy,
                reward_scale=reward_scale,
            )

        print('Memory:')
        print(agent.memory_probs.round(3))
        print()
        print('Policy:')
        print(agent.policy_probs)

        while len(agent.replay) < args.min_mem_opt_replay_size:
            converge_value_functions(agent,
                                     env,
                                     args.n_samples_per_policy,
                                     reward_scale=reward_scale)

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
        'initial_discrep': discrep_start,
        'final_discrep': optim_results['best_discrep'],
        'optim_results': optim_results,
        'initial_policy_history': initial_policy_history,
        'mem_opt_policy_history': mem_opt_policy_history,
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
