import numpy as np

def discount(rewards, gamma):
    gamma_powers = np.arange(len(rewards))
    gamma_multipliers = np.power(gamma, gamma_powers)
    discounted_rewards = np.asarray(rewards) * gamma_multipliers
    backwards_rewards = np.flip(discounted_rewards)
    overdiscounted_returns = np.flip(np.cumsum(backwards_rewards))
    returns = overdiscounted_returns / gamma_multipliers
    return returns

def accumulate_rewards(rewards):
    backwards_rewards = np.flip(np.asarray(rewards))
    return np.flip(np.cumsum(backwards_rewards))

def rollout(mdp, s, a, pi, max_steps=None):
    rewards = []
    oa_pairs = []
    t = 0
    s_t = s
    o_t = mdp.observe(s_t)
    a_t = a
    done = not np.any(mdp.T[a_t][s_t])
    if done:
        raise RuntimeError('Cannot perform rollout with action {} from terminal state {}'.format(
            a_t, s_t))
    while not done:
        next_s, r_t, done = mdp.step(s_t, a_t, mdp.gamma)
        next_obs = mdp.observe(next_s)
        rewards.append(r_t)
        oa_pairs.append((o_t, a_t))
        s_t = next_s
        o_t = next_obs
        a_t = np.random.choice(mdp.action_space.n, p=pi[s_t])
        t += 1
        if max_steps is not None and t >= max_steps:
            break
    # returns = discount(rewards, mdp.gamma)
    returns = accumulate_rewards(rewards)
    mc_returns = [(o, a, g) for ((o, a), g) in zip(oa_pairs, returns)]
    return mc_returns

def mc(mdp,
       pi,
       p0=None,
       alpha=1,
       epsilon=0,
       mc_states='all',
       n_steps=1000,
       max_rollout_steps=None):
    if mc_states not in ['all', 'first']:
        raise ValueError("mc_states must be either 'all' or 'first'")

    # If POMDP, convert to pi_ground
    if hasattr(mdp, 'phi'):
        pi_ground = mdp.get_ground_policy(pi)
    else:
        pi_ground = pi
    if pi_ground.shape[0] != mdp.state_space.n:
        raise ValueError("pi must be a valid policy for the ground mdp")

    if p0 is not None and len(p0) != mdp.state_space.n:
        raise ValueError("p0 must be a valid distribution over ground states")

    q = [np.zeros(mdp.observation_space.n) for _ in range(mdp.action_space.n)]
    for i in range(n_steps):
        s = np.random.choice(mdp.state_space.n, p=p0)
        # use epsilon-greedy action selection for first state
        if np.random.uniform() > epsilon:
            a = np.random.choice(mdp.action_space.n, p=pi_ground[s])
        else:
            a = np.random.choice(mdp.action_space.n)
        mc_returns = rollout(mdp, s, a, pi_ground, max_rollout_steps)

        if i % (n_steps / 10) == 0:
            print(f'Sampling step: {i}/{n_steps}')

        encountered_oa_pairs = set()
        for obs, a, q_target in mc_returns:
            if mc_states == 'all' or (obs, a) not in encountered_oa_pairs:
                q[a][obs] = (1 - alpha) * q[a][obs] + (alpha) * q_target
            encountered_oa_pairs.add((obs, a))

    v = (q * pi.T).sum(0)

    return v, q, pi

def test_discount():
    r = np.arange(5)
    gamma = 0.5
    expected = np.array([
        0 + gamma * 1 + gamma**2 * 2 + gamma**3 * 3 + gamma**4 * 4,
        1 + gamma * 2 + gamma**2 * 3 + gamma**3 * 4,
        2 + gamma * 3 + gamma**2 * 4,
        3 + gamma * 4,
        4,
    ])
    assert all(expected == discount(r, gamma))
