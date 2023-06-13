import argparse
import numpy as np
from pathlib import Path
from time import time

from grl.mdp import POMDP

def generate_pomdps(params):
    timestamp = str(time()).replace('.', '-')
    path = f'grl/environment/pomdp_files/generated/{timestamp}'
    Path(path).mkdir(parents=True, exist_ok=True)

    for i in range(params['n_pomdps']):
        n_s = np.random.randint(params['min_n_s'], params['max_n_s'] + 1)
        n_o = np.random.randint(params['min_n_o'], params['max_n_o'] + 1)
        n_a = np.random.randint(params['min_n_a'], params['max_n_a'] + 1)
        gamma = np.random.random()
        amdp = POMDP.generate(n_s, n_a, n_o, gamma=gamma)

        content = f'# Generation timestamp: {timestamp}\n'
        content += f'# with seed: {args.seed}\n'
        content += f'# with params: {params}\n\n'

        content += f'discount: {amdp.gamma}\n'
        content += 'values: reward\n'
        content += f'states: {amdp.state_space.n}\n'
        content += f'actions: {amdp.action_space.n}\n'
        content += f'observations: {amdp.observation_space.n}\n'
        content += f'start: {str(amdp.p0)[1:-1]}\n\n' # remove array brackets

        # T
        for a in range(amdp.action_space.n):
            content += f'T: {a}\n'
            for row in amdp.T[a]:
                content += f'{str(row)[1:-1]}\n' # remove array brackets

            content += '\n'

        # O
        content += 'O: *\n' # phi currently same for all actions
        for row in amdp.phi:
            content += f'{str(row)[1:-1]}\n' # remove array brackets

        content += '\n'

        # R
        for a in range(amdp.action_space.n):
            for m, row in enumerate(amdp.R[a]):
                for n, val in enumerate(row):
                    content += f'R: {a} : {m} : {n} : * {val}\n'

            content += '\n'

        # Pi_phi
        policies = amdp.generate_random_policies(params['n_policies'])
        for pi in policies:
            content += f'Pi_phi:\n'
            for row in pi:
                content += f'{str(row)[1:-1]}\n' # remove array brackets

            content += '\n'

        with open(f'{path}/{i}.POMDP', 'w') as f:
            f.write(content)

    return timestamp

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_pomdps',
        default=None,
        nargs=8,
        type=int,
        help=
        'args: n_pomdps, n_policies, min_n_s, max_n_s, min_n_a, max_n_a, min_n_o, max_n_o; generate pomdp specs and save to environment/pomdp_files/generated/'
    )
    args = parser.parse_args()
    a = args.generate_pomdps
    params = {
        'n_pomdps': a[0],
        'n_policies': a[1],
        'min_n_s': a[2],
        'max_n_s': a[3],
        'min_n_a': a[4],
        'max_n_a': a[5],
        'min_n_o': a[6],
        'max_n_o': a[7]
    }
    timestamp = generate_pomdps(params)

    print(f'Saved generated pomdp files with timestamp: {timestamp}')
