import numpy as np
import copy


class VoicemailEnv():
    def __init__(self):
        self.start_state_probs=np.array([0.65,0.35])#assuming user wants the massage to be saved
        self.start_state=np.random.multinomial(len(self.start_state_probs),self.start_state_probs).argmax()
        self.current_state=copy.deepcopy(self.start_state)
        self.name="Voicemail "
        self.discount=0.95
        self.reward=0
        self.done=False
        
    def step(self,action):
        if action==0:#refers to asking the user
            transition_probability=np.array([[1,0],[0,1]])
            self.reward=-1
            observation_probability=np.array([[0.8,0.2],[0.3,0.7]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),transition_probability[next_state,:]).argmax()
            self.done=next_state==self.current_state
        
        
        if action==1:#refers to Saving the data
            transition_probability=np.array([[0.65,0.35],[0.65,0.35]])
            rewards=np.array([5,-10])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),transition_probability[next_state,:]).argmax()
            self.reward=rewards[self.current_state]
            self.done=next_state==self.current_state
        
        
        if action==2:#refers to deleting the data
            transition_probability=np.array([[0.65,0.35],[0.65,0.35]])
            rewards=np.array([-20,5])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),transition_probability[next_state,:]).argmax()
            self.reward=rewards[self.current_state]
            self.done=next_state==self.current_state
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
        
            
            