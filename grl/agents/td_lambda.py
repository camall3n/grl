import numpy as np

class TDLambdaQFunction:
    def __init__(self,
                 n_observations: int,
                 n_actions: int,
                 lambda_: float,
                 gamma: float = 0.99,
                 learning_rate: float = 1,
                 trace_type: str = 'accumulating') -> None:
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.lambda_ = lambda_
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.trace_type = trace_type

        self._reset_q_values()
        self._reset_eligibility()

    def _reset_q_values(self):
        self.q = np.zeros((self.n_actions, self.n_observations))

    def _reset_eligibility(self):
        self.eligibility = np.zeros((self.n_actions, self.n_observations))

    def update(self, obs, action, reward, terminal, next_obs, next_action):
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
        self.eligibility *= self.lambda_
        if self.trace_type == 'accumulating':
            self.eligibility[action, obs] += 1
        elif self.trace_type == 'replacing':
            self.eligibility[action, obs] = 1
        else:
            raise RuntimeError(f'Unknown trace_type: {self.trace_type}')
        delta = reward + self.gamma * self.q[next_action, next_obs] - self.q[action, obs]
        self.q += self.learning_rate * delta * self.eligibility

        if terminal:
            self._reset_eligibility()

def run_td_lambda_on_mdp(
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

    print(f"Running TD(λ) with λ = {lambda_}")

    tdlq = TDLambdaQFunction(mdp.n_obs, mdp.n_actions, lambda_, mdp.gamma, alpha)

    for i in range(n_episodes):
        z = np.zeros((mdp.n_actions, mdp.n_obs)) # eligibility traces
        s = np.random.choice(mdp.n_states, p=mdp.p0)
        a = np.random.choice(mdp.n_actions, p=pi_ground[s])
        done = False
        while not done:
            next_s, r, done = mdp.step(s, a, mdp.gamma)
            next_a = np.random.choice(mdp.n_actions, p=pi_ground[next_s])
            ob = mdp.observe(s)
            next_ob = mdp.observe(next_s)

            tdlq.update(ob, a, r, done, next_ob, next_a)

            s = next_s
            a = next_a

        if i % (n_episodes / 10) == 0:
            print(f'Sampling episode: {i}/{n_episodes}')

    q = tdlq.q
    v = (q * pi.T).sum(0)
    return v, q
