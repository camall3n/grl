import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
from tqdm import tqdm

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory

which_tmaze = '5'
for env_name in [
    f'tmaze_{which_tmaze}_two_thirds_up',
    # 'tiger-alt-start',
    # 'cheese.95',
    # 'paint.95',
    # 'network',
    # 'shuttle.95',
    # '4x3.95',
]:
    print(env_name)
    for seed in tqdm(range(1,11)):
        # try:
        # env_name = '
        # study_name = f'{env_name}/exp03-mi0{seed}'
        study_name = f'exp09-tmaze{which_tmaze}-startup/{env_name}/{seed}'
        # study_name = f'tmaze5_tpe_1k'

        spec = environment.load_spec(env_name, memory_id=None)
        mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
        env = AbstractMDP(mdp, spec['phi'])
        agent = ActorCritic(n_obs=env.n_obs,
                            n_actions=env.n_actions,
                            gamma=env.gamma,
                            n_mem_entries=0,
                            replay_buffer_size=int(4e6))
        agent.reset_policy()
        # agent.set_policy(spec['Pi_phi'][0], logits=False) # policy over non-memory observations
        agent.add_memory()

        sampler = 'tpe'
        study_dir = f'./results/sample_based/{study_name}'
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(f"{study_dir}/study.journal"))
        study = optuna.load_study(
            study_name=f'{study_name}',
            storage=storage,
        )

        params = [study.best_trial.params[key] for key in sorted(study.best_trial.params.keys(), key=int)]
        agent.set_memory(agent.fill_in_params(params), logits=False)
        # print(agent.cached_memory_fn.round(3))
        mem = agent.cached_memory_fn.round(3)
        print(str(np.concatenate((mem[2, 0], mem[2, 1], mem[2, 2]), axis=-1)))
        print()

        plot_optimization_history(study)
        plt.title(f'Optimization History ({sampler.upper()})')
        plt.xlim([-20, 4000])
        plt.ylim([-0.01, 1.0])
        plt.legend(loc='upper left', facecolor='white', framealpha=0.9)
        plt.savefig(f'{study_dir}/opt-history-{sampler}.png')
        plt.close()

        # except:
        #     continue
