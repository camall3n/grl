import glob
import os

import numpy as np
from numpy.linalg import norm
import optuna
import pandas as pd

def main():
    env_name = 'tmaze_5_two_thirds_up'
    glob_pattern = 'tpe50k'

    results_dirs = glob.glob(f'results/sample_based/{glob_pattern}')
    output_file = f'results/sample_based/{glob_pattern.replace("*", "")}-{env_name}-summary.pkl'

    assert not os.path.exists(
        output_file), 'Output file already exists. Disable this assert to overwrite it.'

    studies = []
    for results_dir in results_dirs:
        experiment_name = os.path.basename(results_dir)
        envs = sorted(list(map(os.path.basename, glob.glob(f'{results_dir}/*'))))
        if env_name not in envs:
            continue
        seeds = sorted(list(map(os.path.basename, glob.glob(f'{results_dir}/{env_name}/*'))))
        for seed in seeds:
            study = load_study_into_pandas(experiment_name, env_name, seed)
            studies.append(study)
    if studies == []:
        raise RuntimeError(f'No results for "{env_name}" in files matching "{glob_pattern}"')
    data = pd.concat(studies, ignore_index=True)

    data.to_pickle(output_file)

def extract_tmaze_mem_coords(trial):
    p_m0_after_m0_o0 = trial.params['20'] # y
    p_m0_after_m0_o1 = trial.params['22'] # y
    p_m0_after_m0_o2 = trial.params['24'] # p1
    p_m0_after_m1_o2 = trial.params['25'] # p2

    p_initial = np.array([p_m0_after_m0_o0, p_m0_after_m0_o1])
    p0 = np.minimum(norm(p_initial - np.array([0, 1])), norm(p_initial - np.array([1, 0])))
    return p0, p_m0_after_m0_o2, p_m0_after_m1_o2

def load_study(experiment_name, env_name, seed):
    study_name = f'{experiment_name}/{env_name}/{seed}'
    study_dir = f'./results/sample_based/{study_name}'
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(f"{study_dir}/study.journal"))
    study = optuna.load_study(
        study_name=f'{study_name}',
        storage=storage,
    )
    return study

def load_study_into_pandas(experiment_name, env_name, seed):
    study = load_study(experiment_name, env_name, seed)
    df = pd.DataFrame([
        {
            'trial_id': t.number,
            'experiment_name': experiment_name,
            'env_name': env_name,
            'seed': seed,
            'state': str(t.state).replace('TrialState.', ''),
            'mem_coords': extract_tmaze_mem_coords(t),
            'value': t.values[0] if t.values is not None and t.values != [] else np.nan,
            # 'datetime_start': t.datetime_start,
            # 'datetime_complete': t.datetime_complete,
            # 'duration': (t.datetime_complete - t.datetime_start) if (t.datetime_complete is not None) else None,
        } for t in study.trials if str(t.state) == 'TrialState.COMPLETE'
    ])
    complete_trials = df.query('state=="COMPLETE"')
    values_by_trial_id = complete_trials.sort_values(by='trial_id')['value'].tolist()
    best_value_by_trial_id = np.minimum.accumulate(values_by_trial_id)
    complete_trials = complete_trials.assign(best_value=best_value_by_trial_id)
    return complete_trials

if __name__ == '__main__':
    main()
#%%

# data = pd.read_csv('analytical_tmaze_plot_data.csv')
# data.to_pickle('analytical_tmaze_plot_data.pkl')
