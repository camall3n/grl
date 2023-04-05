"""
Script to convert hyperparams/XYZ.py files into a .txt file
where every line of the .txt file is one experiment.
"""
import argparse
import numpy as np
import importlib.util
from typing import List
from pathlib import Path
from itertools import product

from definitions import ROOT_DIR

def import_module_to_hparam(hparam_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("hparam", hparam_path)
    hparam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hparam_module)
    hparams = hparam_module.hparams
    return hparams

def generate_runs(run_dicts: List[dict],
                  runs_dir: Path,
                  experiment_name: str = None,
                  runs_fname: str = 'runs.txt',
                  main_fname: str = 'main.py') -> None:
    """
    :param run_dicts: A list of dictionaries, each specifying a job to run.
    :param runs_dir: Directory to put the runs
    :param runs_fname: What do we call our run file?
    :param main_fname: what is our python entry script?
    :return: nothing. We write to runs_dir/runs_fname
    """

    runs_path = runs_dir / runs_fname

    if runs_path.is_file():
        runs_path.unlink()

    f = open(runs_path, 'a+')

    num_runs = 0
    for run_dict in run_dicts:
        keys, values = [], []

        for k, v in run_dict.items():
            keys.append(k)
            if not (isinstance(v, list) or isinstance(v, np.ndarray)):
                v = [v]
            values.append(v)

        for i, args in enumerate(product(*values)):

            arg = {k: v for k, v in zip(keys, args)}

            run_string = f"python {main_fname}"

            for k, v in arg.items():

                if v is True:
                    run_string += f" --{k}"
                elif v is False or v is None:
                    continue
                else:
                    run_string += f" --{k} {v}"

            if experiment_name is not None:
                run_string += f" --study_name {experiment_name}"

            run_string += "\n"
            f.write(run_string)
            num_runs += 1

            print(num_runs, run_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparam', default='', type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    hparam_path = Path(ROOT_DIR, 'scripts', 'hyperparams', args.hparam + ".py")
    hparams = import_module_to_hparam(hparam_path)

    results_dir = Path(ROOT_DIR, 'results')
    # if not args.local:
    #     # Here we assume we want to write to the scratch directory in CC.
    #     results_dir = Path("/home/taodav/scratch/uncertainty/results")

    # Make our directories if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    main_fname = '-m grl.run'
    if 'entry' in hparams:
        main_fname = hparams['entry']

    pairs = None
    if 'pairs' in hparams:
        pairs = hparams['pairs']

    generate_runs(hparams['args'],
                  runs_dir,
                  runs_fname=hparams['file_name'],
                  main_fname=main_fname,
                  experiment_name=args.hparam)

    print(f"Runs wrote to {runs_dir / hparams['file_name']}")
