from grl.environment import load_pomdp
from grl.utils.loss import discrep_loss


if __name__ == "__main__":
    pomdp, info = load_pomdp('four_tmaze_two_thirds_up')
    pi = info['Pi_phi'][0]

    discrep = discrep_loss(pi, pomdp)

    print()
