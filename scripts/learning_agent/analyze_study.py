import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

study_name = 'tmaze_cmaes_2k'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./results/sample_based/{study_name}/study.journal"))
study = optuna.load_study(
    study_name=f'{study_name}',
    storage=storage,
)

plot_contour(study)
plot_edf(study)
plot_optimization_history(study)
plot_parallel_coordinate(study)
plot_param_importances(study)
plot_slice(study)
