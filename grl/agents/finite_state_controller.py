import numpy as np
from tqdm import tqdm

from grl.agents.base_agent import BaseAgent
from grl.utils.math import glorot_init, softmax

class Node:
    def __init__(self,
                 n_actions: int,) -> None:
        """
        :param n_actions: number of actions
        
        TODO: need this class?
        """
        self.n_actions = n_actions

        # Action probability distribution in this node
        self.psi = glorot_init(n_actions)

    def select_action(self):
        psi = self.psi[np.newaxis, :]
        return np.random.choice(self.n_actions, p=softmax(psi)[0])


class FiniteStateController(BaseAgent):
    """
    Based on "Learning Finite-State Controllers for Partially Observable Environments"
        by Meuleau, Peshkin, Kim, and Kaelbling
    
    Maintains a policy graph.

    psi(n, a) - P(a|n)
    eta(n, o, n') - P(n'|o, n)
    eta_0(o, n) - P(n|o)

    The parameters/weights, w_k, for the functions psi, eta, and eta_0 are "Q-values",
    real numbers that are softmaxed to get the probability distributions.
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
        :param eta_0: (OxN); "Q-value" (weight) for each (initial) obs and (initial) node
        :param alpha: learning rate
        """

        super().__init__(replay_buffer_size)
        self.n_obs = n_obs
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.eta_0 = eta_0 if eta_0 is not None else glorot_init([self.n_obs, self.n_nodes])
        self.alpha = alpha
        assert self.eta_0.shape == (n_obs, n_nodes)
        self.reset()

    def reset(self):
        self.nodes = [Node(self.n_actions) for _ in range(self.n_nodes)] # save list of nodes for policy graph
        self.curr_node = None

        # Policy graph transition function (OxNxN)
        self.eta = glorot_init([self.n_obs, self.n_nodes, self.n_nodes])

    def act(self, obs):
        if self.curr_node is None:
            # Select initial node
            # Doing this here because it is conditioned on the first obs
            self.curr_node = np.random.choice(self.n_nodes, p=softmax(self.eta_0)[obs])

        return self.nodes[self.curr_node].select_action()

    def step_memory(self, obs):
        # TODO: add conditioned on actions as an extension to their work
        self.curr_node = np.random.choice(self.n_nodes, p=softmax(self.eta[obs])[self.curr_node])
    