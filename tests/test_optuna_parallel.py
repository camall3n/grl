from multiprocessing import Pool, freeze_support
import os
import shutil
import time
import warnings

import optuna
import numpy as np

from grl.utils.optuna import until_successful

results_dir = "results/test/"

def objective(trial: optuna.Trial):
    print(f'In objective for trial {trial.number}')
    x = until_successful(trial.suggest_float, 'x', low=0.0, high=1.0)
    print(f'Trial {trial.number} suggested float {x}')
    time.sleep(1)
    return x

def build_study(sampler: optuna.samplers.BaseSampler):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        study = optuna.create_study(
            study_name='test_optuna_parallel',
            direction='minimize',
            storage=optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(os.path.join(results_dir, "study.journal"))),
            sampler=sampler,
            load_if_exists=True,
        )
    return study

def worker(seed, n_trials):
    print(f'in worker {seed}')
    print(f'optimizing study {seed}')
    study = build_study(
        optuna.samplers.TPESampler(n_startup_trials=100, constant_liar=True, seed=seed))
    study.optimize(objective, n_trials=n_trials)

def build_study_and_launch_workers(n_trials=100, n_jobs=12):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    build_study(optuna.samplers.TPESampler(n_startup_trials=100, constant_liar=True))

    n_jobs = max(n_jobs, 1)
    n_trials_per_worker = list(map(len, np.array_split(np.arange(n_trials), n_jobs)))
    worker_seeds = np.arange(n_trials)
    worker_args = zip(worker_seeds, n_trials_per_worker)

    print(f'Starting pool with {n_jobs} workers')
    print(f'n_trials_per_worker: {1}')

    freeze_support()
    print('froze support, whatever that means')
    pool = Pool(n_jobs, maxtasksperchild=1) # Each new tasks gets a fresh worker
    print('made a pool. yay')
    pool.starmap(worker, worker_args)
    print('starmap made')
    pool.close()
    print('pooltime is over')
    pool.join()
    print('join me')

if __name__ == '__main__':
    build_study_and_launch_workers()
