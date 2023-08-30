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
        oi = info['episodes_info']['online_info']
        if args.gamma is None:
            env = get_env(args)
            args.gamma = env.gamma
            raise NotImplementedError('Hyperparam gamma is None. Load the environment?')

        returns = []
        discounted_returns = []

        for ep_len, ep_common_reward, ep_compressed_rewards in \
            zip(oi['episode_length'], oi['most_common_reward'], oi['compressed_rewards']):
            ep_rewards = np.array(uncompress_episode_rewards(ep_len, ep_common_reward, ep_compressed_rewards))
            returns.append(ep_rewards.sum())
            discounted_returns.append(np.dot(ep_rewards, args.gamma ** np.arange(len(ep_rewards))))

        returns = np.array(returns, dtype=np.float16)
        discounted_returns = np.array(discounted_returns, dtype=np.float16)

        del oi['episode_length']
        del oi['most_common_reward']
        del oi['compressed_rewards']

        oi['episode_returns'] = returns
        oi['discounted_returns'] = discounted_returns
        oi['total_episode_loss'] = np.array(oi['total_episode_loss'], dtype=np.float16)
        # res_path.unlink(missing_ok=True)
        np.save(res_path, info)



