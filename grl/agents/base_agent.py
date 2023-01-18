from grl.agents.replaymemory import ReplayMemory

class BaseAgent:
    def __init__(self,
                 replay_buffer_size: int):
        self.replay = ReplayMemory(capacity=replay_buffer_size)

    def store(self, experience):
        self.replay.push(experience)

    def act(self, obs):
        pass
        
    def step_memory(self, obs, action):
        pass
        