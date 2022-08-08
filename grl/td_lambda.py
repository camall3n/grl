import numpy as np

def td_lambda(
    mdp,
    pi,
    lambda_=1,
    alpha=1,
    n_episodes=1000,
):
    # If AMDP, convert to pi_ground
    if hasattr(mdp, 'phi'):
        pi_ground = mdp.get_ground_policy(pi)
    else:
        pi_ground = pi
    if pi_ground.shape[0] != mdp.n_states:
        raise ValueError("pi must be a valid policy for the ground mdp")

    q = np.zeros((mdp.n_actions, mdp.n_obs))

    for i in range(n_episodes):
        z = np.zeros((mdp.n_actions, mdp.n_obs)) # eligibility traces
        s = np.random.choice(mdp.n_states, p=mdp.p0)
        done = False
        while not done:
            a = np.random.choice(mdp.n_actions, p=pi_ground[s])
            next_s, r, done = mdp.step(s, a, mdp.gamma)
            ob = mdp.observe(s)
            next_ob = mdp.observe(next_s)

            # Because mdp.step() terminates with probability (1-γ),
            # we have already factored in the γ that we would normally
            # use to decay the eligibility.
            #
            # The fact that we've arrived at this state and we want to
            # compute the update here means we're in a trajectory where
            # we didn't terminate w.p. (1-γ); rather, we continued with
            # probability γ.
            #
            # Thus we simply decay eligibility by λ.
            z *= lambda_
            z[a, ob] += 1 # Accumulating traces
            # z[a, ob] = 1 # Replacing traces
            delta = r + mdp.gamma * q[a, next_ob] - q[a, ob]
            q += alpha * delta * z

            s = next_s

        if i % (n_episodes / 10) == 0:
            print(f'Sampling episode: {i}/{n_episodes}')

    v = (q * pi.T).sum(0)
    return v, q
