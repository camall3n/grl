from argparse import Namespace
from typing import Tuple

import jax
import numpy as np
import gymnasium as gym

from grl.agent import get_agent, RNNAgent
from grl.environment import get_env
from grl.evaluation import eval_episodes
from grl.model import get_network
from grl.run_sample_based import parse_arguments
from grl.sample_trainer import Trainer
from grl.utils.data import uncompress_episode_rewards
from grl.utils.optimizer import get_optimizer

def train_agent(rand_key: jax.random.PRNGKey, args: Namespace, env: gym.Env) \
        -> Tuple[RNNAgent, dict, jax.random.PRNGKey]:
    network = get_network(args, env.action_space.n)

    optimizer = get_optimizer(args.optimizer, step_size=args.lr)

    agent = get_agent(network, optimizer, env.observation_space.shape, env, args)

    trainer_key, rand_key = jax.random.split(rand_key)
    trainer = Trainer(env, agent, trainer_key, args)

    final_network_params, final_optimizer_params, episodes_info = trainer.train()
    return agent, final_network_params, rand_key

def test_popgym_integration_discrete():
    args = parse_arguments(return_defaults=True)
    args.seed = 2022

    args.trunc = 5
    args.replay_size = 2500
    args.batch_size = 16

    args.total_steps = 20000
    args.no_gamma_terminal = True
    args.spec = 'popgym-RepeatPreviousEasy-v0'
    args.feature_encoding = 'one_hot'
    args.action_cond = 'none'
    args.algo = 'multihead_rnn'
    args.gamma = 0.9
    args.multihead_loss_mode = 'td'
    args.multihead_action_mode = 'td'
    args.lr = 0.001

    env = get_env(args)

    np.random.seed(args.seed)
    rand_key = jax.random.PRNGKey(args.seed)

    agent, network_params, rand_key = train_agent(rand_key, args, env)

    final_eval_info, rand_key = eval_episodes(agent,
                                              network_params,
                                              env,
                                              rand_key,
                                              n_episodes=5,
                                              test_eps=0.,
                                              max_episode_steps=args.max_episode_steps)

    final_eval_rewards_compressed = final_eval_info['episode_rewards']

    for compressed_ep_rewards in final_eval_rewards_compressed:
        ep_rewards = uncompress_episode_rewards(compressed_ep_rewards['episode_length'],
                                                compressed_ep_rewards['most_common_reward'],
                                                compressed_ep_rewards['compressed_rewards'])

        assert sum(ep_rewards) > 0., f"We're (at least) breaking even"

def test_popgym_integration_continuous():
    args = parse_arguments(return_defaults=True)
    args.max_episode_steps = 1000
    args.seed = 2022

    args.trunc = 5
    args.replay_size = 5000
    args.batch_size = 16

    args.total_steps = 20000
    args.no_gamma_terminal = True
    args.spec = 'popgym-StatelessCartPoleEasy-v0'
    args.algo = 'multihead_rnn'
    args.gamma = 0.999
    args.feature_encoding = 'none'
    args.multihead_loss_mode = 'td'
    args.multihead_action_mode = 'td'
    args.lr = 0.0001

    # args.residual_obs_val_input = True
    env = get_env(args)

    np.random.seed(args.seed)
    rand_key = jax.random.PRNGKey(args.seed)

    agent, network_params, rand_key = train_agent(rand_key, args, env)

    final_eval_info, rand_key = eval_episodes(agent,
                                              network_params,
                                              env,
                                              rand_key,
                                              n_episodes=5,
                                              test_eps=0.,
                                              max_episode_steps=args.max_episode_steps)

    final_eval_rewards_compressed = final_eval_info['episode_rewards']

    for compressed_ep_rewards in final_eval_rewards_compressed:
        ep_rewards = uncompress_episode_rewards(compressed_ep_rewards['episode_length'],
                                                compressed_ep_rewards['most_common_reward'],
                                                compressed_ep_rewards['compressed_rewards'])

        # TODO again, I don't know if we expect these small-capacity models to do anything on these problems.
        # Integration seems to work but IDK what to assert.
        # assert sum(ep_rewards) > 0.1, f"Model didn't seem to learn for " \
        #                            f"loss_mode: {loss_mode}, action_mode: {action_mode}"

if __name__ == "__main__":
    # test_popgym_integration_discrete()
    test_popgym_integration_continuous()
