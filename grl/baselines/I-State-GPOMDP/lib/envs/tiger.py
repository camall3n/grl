import numpy as np
import copy

class TigerEnv():
    def __init__(self):
        self.start_state_prob=np.array([0.5,0.5])
        self.start_state=np.random.multinomial(len(self.start_state_prob),self.start_state_prob).argmax()
        self.current_state=copy.deepcopy(self.start_state)
        self.name="Tiger"
        self.discount=0.95
        self.reward=0
        self.done=False
        
    def step(self,action):
     
        
        if action==0: ##corresponds to listen action
            self.reward=-1
            transition_probability=np.array([[1,0],[0,1]])
            observation_probability=np.array([[0.85,0.15],[0.15,0.85]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.done=next_state==self.current_state
           
        if action==1: ##open left
            rewards=np.array([[-100],[10]])
            transition_probability=np.array([[0.5,0.5],[0.5,0.5]])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.reward=rewards[self.current_state]
            self.done=next_state==self.current_state

           
            
        if action==2: ##open right 
            rewards=np.array([[10],[-100]])
            transition_probability=np.array([[0.5,0.5],[0.5,0.5]])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.reward=rewards[self.current_state]
            done=next_state==self.current_state
            
            
        self.current_state=next_state
        return observation,self.reward,self.done
    def number_of_actions(self):
        actions=3
        return actions
    def number_of_observations(self):
        observations=2
        return observations
    def reset(self):
        self.current_state=copy.deepcopy(self.start_state)
        return self.current_state