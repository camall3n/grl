from functools import partial

import numpy as np
import jax
from jax.config import config
import jax.numpy as jnp
from tqdm import tqdm

from grl.environment import load_spec
from grl.agent.actorcritic import ActorCritic
from grl.mdp import POMDP, MDP
from grl.memory import memory_cross_product
from grl.utils.file_system import numpyify_and_save
from grl.utils.policy_eval import lstdq_lambda

from scripts.learning_agent.memory_iteration import parse_args

def featurize_obs_actions(obs: int, action: int, n_obs: int, n_actions: int):
    feature = jnp.zeros(n_obs * n_actions)
    feature = feature.at[obs * (n_obs - 1) + action].set(1)
    return feature

def obs_pi_features(obs: int, pi: jnp.ndarray):
    feature = jnp.zeros_like(pi)
    feature = feature.at[obs].set(pi[obs])
    return jnp.ravel(feature)

@partial(jax.jit, static_argnames=['rew', 'next_obs', 'gamma', 'lambda_'])
def lstdq_update(A: jnp.ndarray, b: jnp.ndarray, z: jnp.ndarray, feature: jnp.ndarray,
                 pi: jnp.ndarray, rew: float, next_obs: int, gamma: float, lambda_: float):
    z = lambda_ * z + feature
    g = obs_pi_features(next_obs, pi)
    A += jnp.outer(z, feature - gamma * g)
    b += rew * z
    return A, b, z

def lstdq_lambda_samples(agent: ActorCritic,
                         env: POMDP,
                         n_samples: int,
                         lambda_: float = 0.,
                         reset_before_converging: bool = False,
                         update_policy: bool = False):
    if reset_before_converging:
        agent.reset_value_functions()

    n_features = env.observation_space.n * env.action_space.n

    A = np.zeros((n_features, n_features))
    b = np.zeros(n_features)

    pbar = tqdm(total=n_samples)
    total_samples = 0
    while total_samples < n_samples:
        agent.reset_memory_state()
        z = np.zeros(n_features)

        obs, _ = env.reset()
        action = agent.act(obs)
        feature = featurize_obs_actions(obs, action, env.observation_space.n, env.action_space.n)

        terminal = False
        eps_length = 0

        while not terminal:
            agent.step_memory(obs, action)
            next_obs, reward, terminal, _, _ = env.step(action)
            eps_length += 1
            next_action = agent.act(next_obs)

            next_feature = featurize_obs_actions(next_obs, next_action, env.observation_space.n,
                                                 env.action_space.n)

            A, b, z = lstdq_update(
                A,
                b,
                z,
                feature,
                agent.policy_probs,
                reward,
                next_obs,
                # gamma=env.gamma * (1 - terminal), lambda_=lambda_)
                gamma=(1 - terminal),
                lambda_=lambda_)
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
            feature = next_feature

        pbar.update(eps_length)
        total_samples += eps_length

    A += np.eye(n_features) * 1e-10
    q_flat = jnp.linalg.solve(A, b)
    q = q_flat.reshape(env.observation_space.n, env.action_space.n).T
    return q

def check_lstdq_samples():
    config.update('jax_platform_name', 'cpu')
    np.set_printoptions(suppress=True)

    args = parse_args()
    args.env = 'tmaze_eps_hyperparams'

    spec = load_spec(args.env, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = POMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]

    agent = ActorCritic(
        n_obs=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=env.gamma,
        n_mem_entries=0,
        policy_epsilon=args.policy_epsilon,
        replay_buffer_size=args.replay_buffer_size,
        # mem_optimizer=args.mem_optimizer,
        # ignore_queue_priority=(not args.enable_priority_queue),
        study_name=f'{args.study_name}/{args.env}/{args.trial_id}',
        use_existing_study=args.use_existing_study,
        discrep_loss=args.discrep_loss,
        disable_importance_sampling=args.disable_importance_sampling,
        override_mem_eval_with_analytical_env=env if args.analytical_mem_eval else None,
    )

    agent.set_policy(pi, logits=False)

    samples_lstd_q = lstdq_lambda_samples(agent, env, 200000, lambda_=0.99999)

    vlstd_lambda_0, qlstd_lambda_0 = lstdq_lambda(agent.policy_probs, env, lambda_=0.99999)
    print("hello")

if __name__ == "__main__":
    check_lstdq_samples()
    # spec = load_spec('tmaze_eps_hyperparams', epsilon=0)
    # mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    # env = AbstractMDP(mdp, spec['phi'])
    #
    # pi = spec['Pi_phi'][0]
    # pi[3, 0] = 1
    # pi[3, 1] = 0
    #
    # feature = featurize_obs_actions(1, 2, env.observation_space.n, env.action_space.n)
    # pi_feature = obs_pi_features(1, pi)
    # print('hello')
