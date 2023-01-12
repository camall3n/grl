import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

from grl import environment
from grl.mdp import AbstractMDP, MDP
from grl.agents.actorcritic import ActorCritic
from grl.environment.memory_lib import get_memory

spec = environment.load_spec('tmaze_2_two_thirds_up', memory_id=None)
mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
env = AbstractMDP(mdp, spec['phi'])
agent = ActorCritic(n_obs=env.n_obs,
                    n_actions=env.n_actions,
                    gamma=env.gamma,
                    n_mem_entries=0,
                    replay_buffer_size=int(4e6))
agent.set_policy(spec['Pi_phi'][0], logits=False) # policy over non-memory observations
agent.add_memory()

study_name = 'tmaze_tpe_2k'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./results/sample_based/{study_name}/study.journal"))
study = optuna.load_study(
    study_name=f'{study_name}',
    storage=storage,
)

params = [study.best_trial.params[key] for key in sorted(study.best_trial.params.keys(), key=int)]
agent.set_memory(agent.fill_in_params(params), logits=False)
print(agent.mem_summary())

#%%
plot_contour(study)
plot_edf(study)
plot_optimization_history(study)
plt.title('Optimization History (TPE)')
plt.ylim([0.05, 0.68])
plot_parallel_coordinate(study)
plot_param_importances(study)
plot_slice(study)
