import argparse

import numpy as np
import jax
from jax.config import config

from grl.agent.lstm import LSTMAgent
from grl.environment import load_spec
from grl.mdp import AbstractMDP, MDP
from grl.model import get_network
from grl.utils.optimizer import get_optimizer
from grl.sample_trainer import Trainer
from grl.utils.file_system import results_path, numpyify_and_save

def parse_arguments():
    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='simple_chain', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--algo', default='lstm', type=str,
                        help='Algorithm to evaluate')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hidden_size', default=10, type=int,
                        help='RNN hidden size')
    parser.add_argument('--trunc', default=-1, type=int,
                        help='RNN truncation length')
    parser.add_argument('--replay_size', default=1, type=int,
                        help='Replay buffer size. Set to 1 for online training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Replay buffer batch size. Set to 1 for online training.')
    parser.add_argument('--action_cond', default=None, type=str,
                        help='Do we do (previous) action conditioning of observations? (None | cat)')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='What epsilon do we use?')
    parser.add_argument('--no_gamma_terminal', action='store_true',
                        help='Do we turn OFF gamma termination?')
    parser.add_argument('--max_episode_steps', default=1000, type=int,
                        help='Maximum number of episode steps')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')
    parser.add_argument('--platform', type=str, default='cpu',
                        help='What platform do we use (cpu | gpu)')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='How often do we checkpoint?')
    parser.add_argument('--total_steps', type=int, default=int(1e4),
                        help='How many total environment steps do we take in this experiment?')
    parser.add_argument('--save_all_checkpoints', action='store_true',
                        help='Do we store ALL of our checkpoints? If not, store only last.')
    parser.add_argument('--seed', default=None, type=int,
                        help='What seed do we use to make the runs deterministic?')
    parser.add_argument('--study_name', type=str, default=None,
                        help='If study name is not specified, we just save things to the root results/ directory.')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    # configs
    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Run
    # Get POMDP definition
    spec = load_spec(args.spec)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])

    results_path = results_path(args)
    all_agents_dir = results_path.parent / 'agent'
    all_agents_dir.mkdir(exist_ok=True)

    agents_dir = all_agents_dir / f'{results_path.stem}'

    network = get_network(args, env.n_actions)

    optimizer = get_optimizer(args.optimizer, step_size=args.lr)

    features_shape = env.observation_space
    if args.action_cond == 'cat':
        features_shape = features_shape[:-1] + (features_shape[-1] + env.n_actions,)

    agent = LSTMAgent(network, optimizer, features_shape, env.n_actions, args)

    trainer = Trainer(env, agent, rand_key, args, checkpoint_dir=agents_dir)

    episodes_info = trainer.train()

    info = {
        'episodes_info': episodes_info,
        'args': vars(args)
    }

    numpyify_and_save(results_path, info)




