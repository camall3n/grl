import argparse
from pathlib import Path

from grl.environment import load_spec
from grl.mdp import MDP
from grl.utils.file_system import numpyify_and_save
from grl.vi import value_iteration

from definitions import ROOT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='example_11', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--tol', default=1e-10, type=float,
                        help='name of POMDP spec; evals Pi_phi policies by default')

    args = parser.parse_args()
    results_path = Path(ROOT_DIR, 'results', 'vi', f'{args.spec}_vi.npy')

    spec = load_spec(args.spec)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    optimal_vs = value_iteration(mdp, tol=args.tol)
    print("Optimal state values from value iteration:")
    print(optimal_vs)
    info = {'optimal_vs': optimal_vs, 'p0': spec['p0'].copy(), 'args': args.__dict__}

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
