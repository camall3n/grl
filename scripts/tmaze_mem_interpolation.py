import jax.numpy as jnp
from jax.nn import softmax
from jax.config import config
from pathlib import Path
from tqdm import tqdm
from functools import partial

from grl.memory.lib import get_memory
from grl.environment import load_spec
from grl.mdp import MDP, POMDP
from grl.memory import memory_cross_product
from grl.utils.math import reverse_softmax
from grl.utils.lambda_discrep import lambda_discrep_measures
from grl.utils.file_system import numpyify_and_save
from grl.utils.loss import discrep_loss, magnitude_td_loss
from definitions import ROOT_DIR

if __name__ == "__main__":
    config.update('jax_platform_name', 'cpu')

    mem_id = 16
    spec_name = 'tmaze_5_two_thirds_up'
    mem_idx = (2, 2, 0)
    save_path = Path(ROOT_DIR, 'results', 'tmaze_mem_interpolation_data.npy')
    # TODO: maybe add discrep_loss_fn variants?
    loss = partial(discrep_loss, value_type='q', error_type='l2', alpha=1.)

    spec = load_spec(spec_name)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = POMDP(mdp, spec['phi'])
    og_mem_params = get_memory(str(mem_id))
    og_mem = softmax(og_mem_params, axis=-1)
    pi = spec['Pi_phi'][0].repeat(2, axis=0)

    assert jnp.isclose(jnp.max(og_mem[mem_idx]), 1)
    argmax_idx = jnp.argmax(og_mem[mem_idx])

    all_res = []

    for fuzz in tqdm(jnp.linspace(0, 1, num=200)):
        og_mem_dist = og_mem[mem_idx].at[1 - argmax_idx].add(fuzz)
        mem_dist = og_mem_dist.at[argmax_idx].add(-fuzz)
        mem = og_mem.at[mem_idx].set(mem_dist)
        mem_aug_amdp = memory_cross_product(reverse_softmax(mem), amdp)
        indv_res = {'fuzz': fuzz, 'mem': mem}
        indv_res.update(lambda_discrep_measures(mem_aug_amdp, pi, discrep_loss_fn=loss))
        all_res.append(indv_res)

    numpyify_and_save(save_path, all_res)
