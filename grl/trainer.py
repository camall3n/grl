from argparse import Namespace
from typing import Union

from grl.mdp import MDP, AbstractMDP
from grl.agent.LSTMAgent import LSTMAgent

class Trainer:
    def __init__(self, env: Union[MDP, AbstractMDP], agent: LSTMAgent,
                 args: Namespace):
        self.env = env
        self.agent = agent
        self.args = args

    def train(self):
        pass
