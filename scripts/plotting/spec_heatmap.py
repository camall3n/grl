import argparse
from functools import partial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from grl.mdp import MDP, POMDP
from grl.environment import load_spec
from grl.utils.loss import discrep_loss

def heatmap(spec: dict, error_type: str = 'l2', num_ticks: int = 5):
    """
    (Currently have to adjust discrep_type and num_ticks above directly)
    """
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])

    # Run for both v and q
    value_types = ['v', 'q']
    for value_type in value_types:

        loss_fn = partial(discrep_loss, value_type=value_type, error_type=error_type)

        discrepancies = []
        x = y = np.linspace(0, 1, num_ticks)
        for i in range(num_ticks):
            p = x[i]
            for j in range(num_ticks):
                q = y[j]
                # TODO: this is not generalizable
                pi = np.array([[p, 1 - p], [q, 1 - q], [0, 0]])
                discrepancies.append(loss_fn(pi, value_type))

                if (num_ticks * i + j + 1) % 10 == 0:
                    print(f'Calculating policy {num_ticks * i + j + 1}/{num_ticks * num_ticks}')

        ax = sns.heatmap(np.array(discrepancies).reshape((num_ticks, num_ticks)),
                         xticklabels=x.round(3),
                         yticklabels=y.round(3),
                         cmap='viridis')
        ax.invert_yaxis()
        ax.set(xlabel='2nd obs', ylabel='1st obs')
        ax.set_title(f'{args.spec}, {value_type}_values, {error_type}_loss')
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',
                        default='example_11',
                        type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--error_type',
                        default='l2',
                        type=str,
                        help='Do we use (l2 | abs) for our discrepancies?')
    parser.add_argument('--num_ticks', default=5, type=int, help='Number of ticks in our heatmap')

    global args
    args = parser.parse_args()
    spec = load_spec(args.spec)

    heatmap(spec, error_type=args.error_type, num_ticks=args.num_ticks)
