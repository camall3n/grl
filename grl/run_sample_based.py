import argparse

import numpy as np
import jax
from jax.config import config

from grl.agent import get_agent
from grl.environment import get_env
from grl.evaluation import eval_episodes
from grl.model import get_network
from grl.sample_trainer import Trainer
from grl.utils.optimizer import get_optimizer
from grl.utils.file_system import results_path, numpyify_and_save
from grl.environment.wrappers import GammaTerminalWrapper

def parse_arguments(return_defaults: bool = False):
    parser = argparse.ArgumentParser()
    # yapf:disable
    # Environment params
    parser.add_argument('--spec', default='simple_chain', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--no_gamma_terminal', action='store_true',
                        help='Do we turn OFF gamma termination?')
    parser.add_argument('--gamma', default=None, type=float,
                        help='Gamma value: overrides environment gamma for our environments, required for Gym environments.')
    parser.add_argument('--max_episode_steps', default=1000, type=int,
                        help='Maximum number of episode steps')
    parser.add_argument('--feature_encoding', default='none', type=str,
                        choices=['one_hot', 'discrete', 'none'],
                        help='What feature encoding do we use?')

    # Agent params
    parser.add_argument('--algo', default='rnn', type=str,
                        choices=['rnn', 'multihead_rnn'],
                        help='Algorithm to evaluate')
    parser.add_argument('--arch', default='gru', type=str,
                        choices=['gru', 'lstm', 'elman'],
                        help='Algorithm to evaluate')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='What epsilon do we use?')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'rmsprop'],
                        help='What optimizer do we use?')

    # RNN hyperparams
    parser.add_argument('--hidden_size', default=10, type=int,
                        help='RNN hidden size')
    parser.add_argument('--value_head_layers', default=0, type=int,
                        help='For our value head network, how deep is it? 0 for a linear value head.')
    parser.add_argument('--trunc', default=-1, type=int,
                        help='RNN truncation length. -1 for the full buffer.')
    parser.add_argument('--action_cond', default="cat", type=str,
                        help='Do we do (previous) action conditioning of observations? (none | cat)')

    # Multihead RNN/Lambda Discrep hyperparams
    parser.add_argument('--multihead_action_mode', default='td', type=str,
                        choices=['td', 'mc'],
                        help='What head to we use for multihead_rnn for action selection?')
    parser.add_argument('--multihead_loss_mode', default='both', type=str,
                        choices=['both', 'td', 'mc'],
                        help='What mode do we use for the multihead RNN loss?')
    parser.add_argument('--residual_obs_val_input', action='store_true',
                        help='For our value function head, do we concatenate obs to the RNN hidden state as input?')
    parser.add_argument('--multihead_lambda_coeff', default=0., type=float,
                        help='What is our coefficient for our lambda discrepancy loss?')
    parser.add_argument('--normalize_rewards', action='store_true',
                        help='Do we normalize our reward range?')

    # Replay buffer hyperparams
    parser.add_argument('--replay_size', default=-1, type=int,
                        help='Replay buffer size. Set to -1 for online training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Replay buffer batch size. Set to 1 for online training.')

    # Logging and checkpointing hyperparams
    parser.add_argument('--offline_eval_freq', type=int, default=None,
                        help='How often do we evaluate offline during training?')
    parser.add_argument('--offline_eval_episodes', type=int, default=1,
                        help='When we do run offline eval, how many episodes do we run?')
    parser.add_argument('--offline_eval_epsilon', type=float, default=None,
                        help='What is our evaluation epsilon? Default is greedy.')
    parser.add_argument('--checkpoint_freq', type=int, default=-1,
                        help='How often do we checkpoint?')
    parser.add_argument('--save_all_checkpoints', action='store_true',
                        help='Do we store ALL of our checkpoints? If not, store only last.')

    # Experiment hyperparams
    parser.add_argument('--total_steps', type=int, default=int(1e4),
                        help='How many total environment steps do we take in this experiment?')
    parser.add_argument('--platform', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='What platform do we use')
    parser.add_argument('--seed', default=None, type=int,
                        help='What seed do we use to make the runs deterministic?')
    parser.add_argument('--study_name', type=str, default=None,
                        help='If study name is not specified, we just save things to the root results/ directory.')

    # For testing: PyTest doesn't like parser.parse_args(), so we just return the defaults.
    if return_defaults:
        defaults = {}
        for action in parser._actions:
            if not action.required and action.dest != "help":
                defaults[action.dest] = action.default
        return argparse.Namespace(**defaults)
    args = parser.parse_args()

    if args.offline_eval_epsilon is None:
        args.offline_eval_epsilon = args.epsilon

    return args

if __name__ == "__main__":
    args = parse_arguments()

    # configs
    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    # config.update('jax_disable_jit', True)

    rand_key = None
    np_rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
        np_rand_state = np.random.RandomState(seed=args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Load environment and env wrappers
    env_key, test_env_key, rand_key = jax.random.split(rand_key, num=3)
    env = get_env(args, rand_state=np_rand_key)
    if not args.no_gamma_terminal:
        env = GammaTerminalWrapper(env, args.gamma)
    test_env = None
    if args.offline_eval_freq is not None:
        test_env = get_env(args, rand_state=np_rand_key)
        # TODO: this is here b/c rock_positions are randomly generated in the env __init__ --
        # refactor this!
        if args.spec == 'rocksample':
            test_env.unwrapped.rock_positions = env.unwrapped.rock_positions.copy()

    results_path = results_path(args)
    all_agents_dir = results_path.parent / 'agent'
    all_agents_dir.mkdir(exist_ok=True)

    agents_dir = all_agents_dir / f'{results_path.stem}'

    network = get_network(args, env.action_space.n)

    optimizer = get_optimizer(args.optimizer, step_size=args.lr)

    obs_size = env.observation_space.shape
    if not obs_size:
        obs_size = (1, )
    agent = get_agent(network, optimizer, obs_size, env, args)

    trainer_key, rand_key = jax.random.split(rand_key)
    trainer = Trainer(env, agent, trainer_key, args, test_env=test_env, checkpoint_dir=agents_dir)

    final_network_params, final_optimizer_params, episodes_info = trainer.train()

    print(f"Finished training. Evaluating over {args.offline_eval_episodes} episodes.")

    final_eval_info, rand_key = eval_episodes(agent, final_network_params, env, rand_key,
                                              n_episodes=args.offline_eval_episodes,
                                              test_eps=args.offline_eval_epsilon,
                                              max_episode_steps=args.max_episode_steps)

    avg_perf = final_eval_info['episode_returns'].mean()

    print(f"Final (averaged) greedy evaluation performance: {avg_perf}")

    info = {
        'episodes_info': episodes_info,
        'args': vars(args),
        'final_eval_info': final_eval_info
    }

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
