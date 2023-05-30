from argparse import Namespace
from typing import Union, Tuple

import jax
import numpy as np

from grl.agent.rnn import RNNAgent
from grl.environment import load_spec
from grl.environment.examples_lib import po_simple_chain
from grl.evaluation import test_episodes
from grl.mdp import MDP, AbstractMDP
from grl.model import get_network
from grl.run_sample_based import parse_arguments
from grl.sample_trainer import Trainer
from grl.utils.data import one_hot
from grl.utils.optimizer import get_optimizer

def train_agent(rand_key: jax.random.PRNGKey, args: Namespace, env: Union[MDP, AbstractMDP]) \
        -> Tuple[RNNAgent, dict]:
    network = get_network(args, env.n_actions)

    optimizer = get_optimizer(args.optimizer, step_size=args.lr)

    features_shape = env.observation_space
    if args.action_cond == 'cat':
        features_shape = features_shape[:-1] + (features_shape[-1] + env.n_actions,)

    agent = RNNAgent(network, optimizer, features_shape, env.n_actions, args)

    trainer_key, rand_key = jax.random.split(rand_key)
    trainer = Trainer(env, agent, trainer_key, args)

    final_network_params, final_optimizer_params, episodes_info = trainer.train()
    return agent, final_network_params

def test_value():
    args = parse_arguments(return_defaults=True)
    chain_length = 10
    args.max_episode_steps = chain_length
    args.seed = 2020
    args.lr = 0.005
    args.total_steps = 3000
    args.no_gamma_terminal = True

    spec = po_simple_chain(n=chain_length)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    rand_key = jax.random.PRNGKey(args.seed)

    agent, network_params = train_agent(rand_key, args, env)

    hs, rand_key = agent.reset(rand_key)

    obs, env_info = env.reset()
    obs = one_hot(obs, env.n_obs)

    # Action conditioning for t=-1 action
    if args.action_cond == 'cat':
        action_encoding = np.zeros(env.n_actions)
        obs = np.concatenate([obs, action_encoding], axis=-1)

    action, rand_key, hs, qs = agent.act(network_params, obs, hs, rand_key)
    action = action.item()

    all_qs = [qs.item()]
    for t in range(args.max_episode_steps):
        next_obs, reward, done, _, info = env.step(action, gamma_terminal=False)
        next_obs = one_hot(next_obs, env.n_obs)

        # Action conditioning
        if args.action_cond == 'cat':
            action_encoding = one_hot(action, env.n_actions)
            next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

        next_action, rand_key, next_hs, qs = agent.act(network_params, next_obs, hs, rand_key)
        next_action = next_action.item()
        all_qs.append(qs.item())

        if done:
            break

        # bump time step
        hs = next_hs
        action = next_action
    all_qs = np.array(all_qs[:-1])

    ground_truth_vals = env.gamma ** (np.arange(chain_length - 1)[::-1])
    mse = ((ground_truth_vals - all_qs) ** 2).mean()

    assert np.isclose(mse, 0, atol=1e-3)

def test_actions():
    args = parse_arguments(return_defaults=True)
    args.max_episode_steps = 1000
    args.seed = 2020
    args.lr = 0.001
    args.trunc = 10
    args.replay_size = 1000
    args.total_steps = 10000
    args.no_gamma_terminal = True
    # args.arch = 'lstm'
    args.spec = 'tmaze_5_two_thirds_up'

    spec = load_spec(args.spec)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    rand_key = jax.random.PRNGKey(args.seed)

    agent, network_params = train_agent(rand_key, args, env)

    final_eval_info, rand_key = test_episodes(agent, network_params, env, rand_key,
                                              n_episodes=5,
                                              test_eps=0., action_cond=args.action_cond,
                                              max_episode_steps=args.max_episode_steps)

    final_eval_rewards = final_eval_info['episode_rewards']

    for ep_rews in final_eval_rewards:
        assert sum(ep_rews) == 4.


if __name__ == "__main__":
    # test_value()
    test_actions()
