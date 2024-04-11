from grl.environment import load_pomdp
from grl.utils.loss import discrep_loss
from grl.vi import value_iteration


if __name__ == "__main__":
    pomdp, info = load_pomdp('four_tmaze_two_thirds_up')
    pi = info['Pi_phi'][0]

    v = value_iteration(pomdp.base_mdp)
    discrep = discrep_loss(pi, pomdp)

    print()
