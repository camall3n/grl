from argparse import Namespace
from typing import Union, Tuple

import jax
import numpy as np

from grl.agent import get_agent, RNNAgent
from grl.environment import get_env
from grl.evaluation import eval_episodes
from grl.mdp import MDP, AbstractMDP
from grl.model import get_network
from grl.run_sample_based import parse_arguments
from grl.sample_trainer import Trainer
from grl.utils.data import uncompress_episode_rewards
from grl.utils.optimizer import get_optimizer

def train_agent(rand_key: jax.random.PRNGKey, args: Namespace, env: Union[MDP, AbstractMDP]) \
        -> Tuple[RNNAgent, dict, jax.random.PRNGKey]:
    network = get_network(args, env.action_space.n)

    optimizer = get_optimizer(args.optimizer, step_size=args.lr)

    agent = get_agent(network, optimizer, env.observation_space.shape, env, args)

    trainer_key, rand_key = jax.random.split(rand_key)
    trainer = Trainer(env, agent, trainer_key, args)

    final_network_params, final_optimizer_params, episodes_info = trainer.train()
    return agent, final_network_params, rand_key

def test_both_values():
    args = parse_arguments(return_defaults=True)
    chain_length = 5
    args.spec = 'po_simple_chain'
    args.max_episode_steps = chain_length
    args.seed = 2020

    args.total_steps = 5000
    args.algo = 'multihead_rnn'
    args.multihead_loss_mode = 'both'
    args.lr = 0.001
    args.no_gamma_terminal = True

    np.random.seed(args.seed)
    rand_key = jax.random.PRNGKey(args.seed)

    env = get_env(args, n=chain_length)

    agent, network_params, rand_key = train_agent(rand_key, args, env)

    hs, rand_key = agent.reset(rand_key)

    obs, env_info = env.reset()

    action, rand_key, next_hs, _ = agent.act(network_params, obs, hs, rand_key)
    new_carry, q_td, q_mc = agent.Qs(network_params,
                                     np.expand_dims(np.expand_dims(obs, 0), 0),
                                     hs,
                                     mode='both')
    hs = next_hs
    action = action.item()

    all_td_qs = [q_td.item()]
    all_mc_qs = [q_mc.item()]
    for t in range(args.max_episode_steps):
        next_obs, reward, done, _, info = env.step(action, gamma_terminal=False)

        next_action, rand_key, next_hs, _ = agent.act(network_params, next_obs, hs, rand_key)
        new_carry, q_td, q_mc = agent.Qs(network_params,
                                         np.expand_dims(np.expand_dims(next_obs, 0), 0),
                                         hs,
                                         mode='both')
        next_action = next_action.item()
        all_td_qs.append(q_td.item())
        all_mc_qs.append(q_mc.item())

        if done:
            break

        # bump time step
        hs = next_hs
        action = next_action

    all_td_qs = np.array(all_td_qs[:-1])
    all_mc_qs = np.array(all_mc_qs[:-1])

    ground_truth_vals = env.gamma**(np.arange(chain_length - 1)[::-1])
    td_mse = ((ground_truth_vals - all_td_qs)**2).mean()
    mc_mse = ((ground_truth_vals - all_mc_qs)**2).mean()

    assert np.isclose(td_mse, 0,
                      atol=1e-3), f'Values are inaccurate for TD value head: {all_td_qs}'
    assert np.isclose(mc_mse, 0,
                      atol=1e-3), f'Values are inaccurate for MC value head: {all_mc_qs}'

def test_gamma_terminal():
    args = parse_arguments(return_defaults=True)
    chain_length = 5
    args.spec = 'po_simple_chain'
    args.max_episode_steps = chain_length
    args.seed = 2020

    args.total_steps = 10000
    args.algo = 'multihead_rnn'

    np.random.seed(args.seed)
    rand_key = jax.random.PRNGKey(args.seed)

    env = get_env(args, n=chain_length)

    args.multihead_loss_mode = 'td'
    args.multihead_action_mode = 'td'
    args.lr = 0.0001 # TD
    # args.lr = 0.0005  # MC

    agent, network_params, rand_key = train_agent(rand_key, args, env)

    final_eval_info, rand_key = eval_episodes(agent,
                                              network_params,
                                              env,
                                              rand_key,
                                              n_episodes=1,
                                              test_eps=0.,
                                              max_episode_steps=args.max_episode_steps)
    all_qs = np.array([q.item() for q in final_eval_info['episode_qs'][0]])[:-1]

    ground_truth_vals = env.gamma**(np.arange(chain_length - 1)[::-1])
    mse = ((ground_truth_vals - all_qs)**2).mean()

    # we set a higher tolerance here b/c gamma termination introduces
    # a lot of instability into the system.
    assert np.isclose(mse, 0,
                      atol=1e-2), f'Values are inaccurate for gamma_terminal TD values: {all_qs}'

def test_td_mc_values():
    args = parse_arguments(return_defaults=True)
    chain_length = 5
    args.spec = 'po_simple_chain'
    args.max_episode_steps = chain_length
    args.seed = 2020

    args.total_steps = 5000
    args.algo = 'multihead_rnn'
    args.no_gamma_terminal = True

    np.random.seed(args.seed)
    rand_key = jax.random.PRNGKey(args.seed)

    env = get_env(args, n=chain_length)

    for mode, lr in zip(['td', 'mc'], [0.0002, 0.001]):
        args.multihead_loss_mode = mode
        args.multihead_action_mode = mode
        args.lr = lr

        agent, network_params, rand_key = train_agent(rand_key, args, env)

        final_eval_info, rand_key = eval_episodes(agent,
                                                  network_params,
                                                  env,
                                                  rand_key,
                                                  n_episodes=1,
                                                  test_eps=0.,
                                                  max_episode_steps=args.max_episode_steps)
        all_qs = np.array([q.item() for q in final_eval_info['episode_qs'][0]])[:-1]

        ground_truth_vals = env.gamma**(np.arange(chain_length - 1)[::-1])
        mse = ((ground_truth_vals - all_qs)**2).mean()

        assert np.isclose(mse, 0, atol=1e-3), f'Values are inaccurate for {mode} values: {all_qs}'

def test_actions():
    args = parse_arguments(return_defaults=True)
    args.max_episode_steps = 1000
    args.seed = 2022

    args.trunc = 5
    args.replay_size = 2500

    args.total_steps = 6000
    args.no_gamma_terminal = True
    args.spec = 'tmaze_hyperparams'
    args.algo = 'multihead_rnn'

    # args.residual_obs_val_input = True

    env = get_env(args, corridor_length=2)

    np.random.seed(args.seed)
    rand_key = jax.random.PRNGKey(args.seed)

    for (loss_mode, action_mode), lr in \
            zip([('td', 'td'), ('mc', 'mc'), ('both', 'td'), ('both', 'mc')], [0.002, 0.0075, 0.0025, 0.005]):
        args.multihead_loss_mode = loss_mode
        args.multihead_action_mode = action_mode
        args.lr = lr

        agent, network_params, rand_key = train_agent(rand_key, args, env)

        final_eval_info, rand_key = eval_episodes(agent,
                                                  network_params,
                                                  env,
                                                  rand_key,
                                                  n_episodes=5,
                                                  test_eps=0.,
                                                  max_episode_steps=args.max_episode_steps)

        final_eval_rewards_compressed = final_eval_info['episode_rewards']

        for compressed_ep_rews in final_eval_rewards_compressed:
            ep_rews = uncompress_episode_rewards(compressed_ep_rews['episode_length'],
                                                 compressed_ep_rews['most_common_reward'],
                                                 compressed_ep_rews['compressed_rewards'])

            assert sum(ep_rews) == 4., f"Optimal actions don't match for " \
                                       f"loss_mode: {loss_mode}, action_mode: {action_mode}"

if __name__ == "__main__":
    # test_gamma_terminal()
    # test_td_mc_values()
    # test_both_values()
    test_actions()
