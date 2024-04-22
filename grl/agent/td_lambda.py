import warnings

import numpy as np

class TDLambdaQFunction:
    def __init__(self,
                 n_obs: int,
                 n_actions: int,
                 lambda_: float,
                 gamma: float = 0.99,
                 learning_rate: float = 1,
                 trace_type: str = 'accumulating') -> None:
        """
        Online implementation of TD(λ)

        :param n_observations:  number of observations
        :param n_actions:       number of actions
        :param lambda_:         lambda value (λ), in range [0, 1] (0 = "TD", 1 = "MC")
        :param gamma:           discount factor, in range [0, 1)
        :param learning_rate:   learning rate / step size parameter, in range [0, 1]
        :param trace_type:      'accumulating' or 'replacing'
        """
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.lambda_ = lambda_
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.trace_type = trace_type

        self.reset()

    def reset(self):
        self._reset_q_values()
        self._reset_eligibility()

    def _reset_q_values(self):
        self.q = np.zeros((self.n_actions, self.n_obs))

    def _reset_eligibility(self):
        self.eligibility = np.zeros((self.n_actions, self.n_obs))

    def update(
            self,
            obs,
            action,
            reward,
            terminal,
            next_obs,
            next_action,
            aug_obs=None,      # memory-augmented observation
            next_aug_obs=None, # and next observation (O x M)
        ):
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
        if aug_obs is not None:
            obs = aug_obs
        if next_aug_obs is not None:
            next_obs = next_aug_obs

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

    def augment_with_memory(self, n_mem_states: int):
        """
        Expand q (A x O) => A x OM to include memory states
        """
        # augment last dim with input mem states
        self.n_obs *= n_mem_states
        # q_mem_states = np.stack((self.q, np.zeros_like(self.q)), axis=-1)
        q_mem_states = np.expand_dims(self.q, -1).repeat(n_mem_states, -1) # A x O x M
        self.q = q_mem_states.reshape(self.n_actions, self.n_obs)

        if np.any(self.eligibility > 0):
            warnings.warn('Resetting non-zero eligibility during memory augmentation')
        self._reset_eligibility()

def run_td_lambda_on_mdp(
    mdp,
    pi,
    lambda_=1,
    alpha=1,
    n_episodes=1000,
):
    # If POMDP, convert to pi_ground
    if hasattr(mdp, 'phi'):
        pi_ground = mdp.get_ground_policy(pi)
    else:
        pi_ground = pi
    if pi_ground.shape[0] != mdp.state_space.n:
        raise ValueError("pi must be a valid policy for the ground mdp")

    print(f"Running TD(λ) with λ = {lambda_}")

    tdlq = TDLambdaQFunction(mdp.observation_space.n, mdp.action_space.n, lambda_, mdp.gamma,
                             alpha)

    for i in range(n_episodes):
        obs, _ = mdp.reset()
        a = np.random.choice(mdp.action_space.n, p=pi[obs])
        done = False
        while not done:
            next_obs, r, done, _, _ = mdp.step(a)
            next_a = np.random.choice(mdp.action_space.n, p=pi[next_obs])

            tdlq.update(obs, a, r, done, next_obs, next_a)

            obs = next_obs
            a = next_a

        if i % (n_episodes / 10) == 0:
            print(f'Sampling episode: {i}/{n_episodes}')

    q = tdlq.q
    v = (q * pi.T).sum(0)
    return v, q
