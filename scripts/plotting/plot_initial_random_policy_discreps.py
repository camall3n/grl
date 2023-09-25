from jax.config import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange

from grl.agent.actorcritic import ActorCritic
from grl.agent.analytical import AnalyticalAgent
from grl.environment import load_spec
from grl.mdp import MDP, POMDP
from grl.utils.loss import discrep_loss
from scripts.learning_agent.memory_iteration import parse_args

data_filepath = 'results/random_policy_discreps/combined_data.pkl'

try:
    data = pd.read_pickle(data_filepath)
except:
    np.set_printoptions(precision=3, suppress=True)
    config.update('jax_platform_name', 'cpu')

    args = parse_args()
    np.random.seed(args.seed)

    args.n_random_policies = 1000000
    del args.f

    envs = [
        'tiger-alt-start',
        'paint.95',
        'tmaze_5_two_thirds_up',
        'shuttle.95',
        'example_7',
        'cheese.95',
        '4x3.95',
    ]

    reward_range_dict = {
        'cheese.95': (10.0, 0),
        'tiger-alt-start': (10.0, -100.0),
        'network': (80.0, -40.0),
        'slippery-tmaze': (4.0, -0.1),
        'tmaze_5_two_thirds_up': (4.0, -0.1),
        'example_7': (1.0, 0.0),
        '4x3.95': (1.0, -1.0),
        'shuttle.95': (10.0, -3.0),
        'paint.95': (1.0, -1.0),
        'bridge-repair': (4018, 0),
        'hallway': (1.0, 0),
    }

    dfs = []

    for env_name in envs:
        reward_scale = 1 / (reward_range_dict[env_name][0] - reward_range_dict[env_name][1])
        spec = load_spec(env_name, memory_id=None)
        mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
        mdp.R *= reward_scale
        env = POMDP(mdp, spec['phi'])

        learning_agent = ActorCritic(
            n_obs=env.observation_space.n,
            n_actions=env.action_space.n,
            gamma=env.gamma,
            lambda_0=args.lambda0,
            lambda_1=args.lambda1,
            learning_rate=args.learning_rate,
            n_mem_entries=0,
            replay_buffer_size=args.replay_buffer_size,
            mem_optimizer=args.mem_optimizer,
            ignore_queue_priority=(not args.enable_priority_queue),
            annealing_should_sample_hyperparams=args.annealing_should_sample_hyperparams,
            annealing_tmax=args.annealing_tmax,
            annealing_tmin=args.annealing_tmin,
            annealing_progress_fraction_at_tmin=args.annealing_progress_fraction_at_tmin,
            n_annealing_repeats=args.n_annealing_repeats,
            prune_if_parent_suboptimal=False,
            mellowmax_beta=10.,
            discrep_loss='mse',
            study_name='compare_sample_and_plan_04/' + args.study_name,
            override_mem_eval_with_analytical_env=env,
            analytical_lambda_discrep_noise=0.00,
        )

        planning_agent = AnalyticalAgent(
            pi_params=learning_agent.policy_probs,
            rand_key=None,
            mem_params=learning_agent.memory_probs,
            value_type='q',
        )

        largest_discrep = 0
        largest_discrep_policy = None
        initial_policy_discreps = [0]
        for i in trange(args.n_random_policies):
            learning_agent.reset_policy()
            policy = learning_agent.policy_probs
            policy_lamdba_discrep = discrep_loss(policy, env)[0].item()
            if policy_lamdba_discrep > largest_discrep:
                largest_discrep = policy_lamdba_discrep
                largest_discrep_policy = policy
            initial_policy_discreps.append(largest_discrep)
        discreps = np.asarray(initial_policy_discreps)
        discreps /= discreps.max()
        series = pd.Series(discreps)
        df = pd.DataFrame({'t': np.arange(len(discreps)), 'discrep': series})
        df['spec'] = env_name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data.to_pickle(data_filepath)

#%%
sns.lineplot(data=data, x='t', y='discrep', hue='spec')
plt.ylabel('Largest normalized discrepancy')
plt.xlabel('Number of policies considered')

#%%
sns.lineplot(data=data, x='t', y='discrep', hue='spec')
plt.ylabel('Largest normalized discrepancy')
plt.xlabel('Number of policies considered')
plt.xlim([-10, 400])
