"""
Converts and runs a specified hyperparams/XYZ.py hyperparam file
into an Onager prelaunch script.
"""
import os
import numpy as np
import argparse
import importlib.util
from typing import List
from pathlib import Path

from definitions import ROOT_DIR

def import_module_to_hparam(hparam_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("hparam", hparam_path)
    hparam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hparam_module)
    hparams = hparam_module.hparams
    return hparams

def generate_onager_runs(run_dicts: List[dict],
                         experiment_name: str,
                         main_fname: str = 'main.py') -> None:
    """
    :param run_dicts: A list of dictionaries, each specifying a job to run.
    Each run_dict in this list corresponds to one call of `onager prelaunch`
    :param main_fname: what is our python entry script?
    :return: nothing. We execute an onager prelaunch script
    """
    onager_prelaunch = "onager prelaunch"
    jobname = f"+jobname {experiment_name}"

    for i, run_dict in enumerate(run_dicts):
        command = f"python {main_fname}"
        if len(run_dicts) > 1:
            jobname += f"_{i}"

        prelaunch_list = [onager_prelaunch, jobname]
        arg_list = []

        for k, v in run_dict.items():
            if not (isinstance(v, list) or isinstance(v, np.ndarray)):
                if isinstance(v, bool):
                    if v:
                        command += f" --{k}"
                    else:
                        continue
                else:
                    command += f" --{k} {v}"
            else:
                if all([isinstance(el, bool) for el in v]):
                    arg_string = f"+flag --{k}"
                else:
                    arg_string = f"+arg --{k} {' '.join(map(str, v))}"
                arg_list.append(arg_string)

        command += f" --study_name {experiment_name}"

        prelaunch_list.append(f'+command "{command}"')
        prelaunch_list += arg_list

        prelaunch_string = ' '.join(prelaunch_list)
        print(f"Launching prelaunch script: {prelaunch_string}")
        os.chdir(ROOT_DIR)
        os.system(prelaunch_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name', default=None, type=str)
    parser.add_argument('--hparam', default='', type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    hparam_path = Path(ROOT_DIR, 'scripts', 'hyperparams', args.hparam + ".py")
    hparams = import_module_to_hparam(hparam_path)

    main_fname = '-m grl.run'
    if 'entry' in hparams:
        main_fname = hparams['entry']

    pairs = None
    if 'pairs' in hparams:
        pairs = hparams['pairs']

    exp_name = args.hparam
    if args.study_name is not None:
        exp_name = args.study_name
    generate_onager_runs(hparams['args'], exp_name, main_fname=main_fname)
