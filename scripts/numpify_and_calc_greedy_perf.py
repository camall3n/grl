import numpy as np
from pathlib import Path
from tqdm import tqdm
from jax.config import config

from grl.utils import load_info, greedify, numpyify_and_save
from grl.policy_eval import lambda_discrep_measures
from grl import load_spec, MDP, AbstractMDP
from grl.memory import memory_cross_product
from definitions import ROOT_DIR

if __name__ == "__main__":
    results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_pi')
    agents_dir = results_dir / 'agents'
    agents_dir.mkdir(exist_ok=True)

    config.update('jax_platform_name', 'cpu')

    for res_path in tqdm(results_dir.iterdir()):
        if res_path.is_dir():
            continue

        info = load_info(res_path)
        if 'agent' not in info:
            continue

        logs = info['logs']
        agent = info['agent']
        args = info['args']

        spec = load_spec(args['spec'], memory_id=args['use_memory'],
                         n_mem_states=args['n_mem_states'],
                         corridor_length=args['tmaze_corridor_length'],
                         discount=args['tmaze_discount'],
                         junction_up_pi=args['tmaze_junction_up_pi'])
        mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
        amdp = AbstractMDP(mdp, spec['phi'])
        final_mem_amdp = memory_cross_product(amdp, agent.mem_params)

        info_less_agent = {k: v for k, v in info.items() if k != 'agent'}

        greedy_init_improvement_policy = greedify(logs['initial_improvement_policy'])
        info_less_agent['logs']['greedy_initial_improvement_stats'] = lambda_discrep_measures(amdp, greedy_init_improvement_policy)

        greedy_final_policy = greedify(agent.policy)
        info_less_agent['logs']['greedy_final_mem_stats'] = lambda_discrep_measures(final_mem_amdp, greedy_final_policy)

        agent_path = agents_dir / f'{res_path.stem}.pkl'

        np.save(agent_path, agent)
        res_path.unlink()
        numpyify_and_save(res_path, info_less_agent)

