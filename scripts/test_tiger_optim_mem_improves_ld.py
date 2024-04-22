from jax.nn import softmax

from grl.environment import load_pomdp
from grl.environment.policy_lib import get_start_pi
from grl.memory import get_memory, memory_cross_product
from grl.utils.loss import discrep_loss
from grl.utils.math import softmax

pomdp, pi_dict = load_pomdp('tiger-alt-start', n_mem_states='2')
pi_params = get_start_pi('tiger_alt_start_known_ld', pi_phi=None)

initial_discrep = discrep_loss(softmax(pi_params), pomdp)[0].item()
print(f'initial_discrep = {initial_discrep}')

mem_params = get_memory('tiger_alt_start_1bit_optimal',
                        n_obs=pomdp.observation_space.n,
                        n_actions=pomdp.action_space.n,
                        n_mem_states=2)

final_pomdp = memory_cross_product(mem_params, pomdp)
aug_pi_params = pi_params.repeat(2, axis=0)
final_discrep = discrep_loss(softmax(aug_pi_params), final_pomdp)[0].item()
print(f'final_discrep = {final_discrep}')
