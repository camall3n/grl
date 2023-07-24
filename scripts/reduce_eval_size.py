from pathlib import Path
import argparse
from argparse import Namespace

import numpy as np
from tqdm import tqdm

from grl.environment import get_env
from grl.utils.data import uncompress_episode_rewards
from grl.utils.file_system import load_info, numpyify_and_save
from grl.utils.loss import mse
from grl.utils.mdp import all_t_discounted_returns

from definitions import ROOT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_name', default='popgym_sweep_mc', type=str,
                        help='Results directory in the results folder to parse.')

    args = parser.parse_args()

    results_dir = Path(ROOT_DIR, 'results', args.dir_name)

    all_results_dir = [res_path for res_path in results_dir.iterdir() if res_path.suffix == '.npy']

    for res_path in tqdm(all_results_dir):
        info = load_info(res_path)
        args = Namespace(**info['args'])
        offline_evals = info['episodes_info']['offline_eval']
        if args.gamma is None:
            env = get_env(args)
            args.gamma = env.gamma
            raise NotImplementedError('Hyperparam gamma is None. Load the environment?')

        new_offline_evals = []

        for oe in offline_evals:
            to_save = {'returns': [], 'q_err': []}
        
            if args.no_gamma_terminal:
                to_save['discounted_returns'] = []

            # iterate through number of seeds
            for ep_rewards, ep_qs in zip(oe['episode_rewards'], oe['episode_qs']):
                # relevant qs are everything but the last index
                # b/c last index is terminal q
                relevant_ep_qs = ep_qs[:-1]

                episode_rewards = np.array(uncompress_episode_rewards(ep_rewards['episode_length'], ep_rewards['most_common_reward'], ep_rewards['compressed_rewards']))

                discounts = np.ones(episode_rewards.shape[0])
                if args.no_gamma_terminal:
                    discounts *= args.gamma

                discounts[-1] = 0.

                t_discounted_returns = all_t_discounted_returns(discounts, episode_rewards)

                to_save['q_err'].append(mse(relevant_ep_qs, t_discounted_returns).item())
                to_save['returns'].append(episode_rewards.sum())
                if args.no_gamma_terminal:
                    to_save['discounted_returns'].append(t_discounted_returns[0])

            new_offline_evals.append(to_save)

        info['episodes_info']['offline_eval'] = new_offline_evals
        # res_path.unlink(missing_ok=True)
        np.save(res_path, info)



