import numpy as np
from tqdm import tqdm

from grl.agents.base_agent import BaseAgent
from grl.utils.math import glorot_init, softmax, softmax_derivative

class FiniteStateController(BaseAgent):
    """
    Based on "Learning Finite-State Controllers for Partially Observable Environments"
        by Meuleau, Peshkin, Kim, and Kaelbling
    
    Maintains a policy graph.

    psi(n, a) - P(a|n)
    eta(n, o, n') - P(n'|o, n)
    eta_0(o, n) - P(n|o)

    The parameters/weights, w_k, for the functions psi, eta, and eta_0 are logits referred to as "Q-values".
    TODO: need temperature parameter?
    """
    def __init__(self,
                 n_obs: int,
                 n_actions: int,
                 n_nodes: int,
                 eta_0: np.ndarray=None,
                 replay_buffer_size: int = 1000000,
                 alpha=0.1) -> None:
        """
        :param n_obs: number of observations
        :param n_actions: number of actions
        :param n_nodes: number of nodes in policy graph
        :param eta_0: (OxN); "Q-value" (logit) for each (initial) obs and (initial) node
        :param alpha: learning rate
        """

        super().__init__(replay_buffer_size)
        self.n_obs = n_obs
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.alpha = alpha


        self.eta_0 = eta_0 if eta_0 is not None else glorot_init([self.n_obs, self.n_nodes])
        assert self.eta_0.shape == (n_obs, n_nodes)
        self.reset()

        # Stuff for self.update_weights()
        self.first_step = True # Track first step in episode for gradient: eta_0 or eta

    def reset(self):
        self.curr_node = None

        # Action logits for each node (NxA)
        self.psi = glorot_init([self.n_nodes, self.n_actions])

        # Policy graph transition function (OxNxN)
        self.eta = glorot_init([self.n_obs, self.n_nodes, self.n_nodes])

        self.reset_gradients()

    def reset_gradients(self):
        self.grad_psi = np.zeros_like(self.psi)
        self.grad_eta = np.zeros_like(self.eta)
        self.grad_eta_0 = np.zeros_like(self.eta_0)

    def act(self, obs):
        if self.curr_node is None:
            # Select initial node
            # Doing this here because it is conditioned on the first obs
            self.curr_node = np.random.choice(self.n_nodes, p=softmax(self.eta_0)[obs])

        psi = self.psi[self.curr_node][np.newaxis, :] # Make 2D for softmax
        action = np.random.choice(self.n_actions, p=softmax(psi)[0])
        self.step_memory(obs, action)
        return action

    def step_memory(self, obs, action):
        # TODO: add conditioned on actions as an extension to their work
        self.curr_node = np.random.choice(self.n_nodes, p=softmax(self.eta[obs])[self.curr_node])

    
    def update_weights(self, experience):
        """
        Used to update weights at end of each trial.
        Calculates and sums incremental (per-step) updates and applies at end of trial.
        """
        curr_node = experience['curr_node']
        next_node = experience['next_node']
        action = experience['action']
        obs = experience['obs']
        reward = experience['reward']

        #TODO: these are wrong; some weights change but not converging

        self.grad_psi[curr_node] -= (self.alpha * 
                                     reward * 
                                     softmax_derivative(np.log(softmax(self.psi[curr_node][np.newaxis, :])[0]))[action])

        if self.first_step:
            #TODO: doesn't update unless there's non-zero reward in the first step; is that right? should it be updated at every step? return?
            self.grad_eta_0[obs] -= (self.alpha * 
                                     reward * 
                                     softmax_derivative(np.log(softmax(self.eta_0[obs][np.newaxis, :])[0]))[curr_node])
            self.first_step = False

        self.grad_eta[obs,curr_node] -= (self.alpha * 
                                         reward * 
                                         softmax_derivative(np.log(softmax(self.eta[obs,curr_node][np.newaxis, :])[0]))[next_node])


        if experience['terminal'] == True:
            self.psi += self.grad_psi
            self.eta_0 += self.grad_eta_0
            self.eta += self.grad_eta

            print('grad_eta0', self.grad_eta_0)
            print('eta0', self.eta_0)
            print('PSI', self.psi)
            print('eta', self.eta)

            self.reset_gradients()
            self.first_step = True
