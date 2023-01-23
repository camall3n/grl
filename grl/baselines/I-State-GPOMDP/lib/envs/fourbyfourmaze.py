import numpy as np
import copy



class FourByFourMazeEnv():
    def __init__(self):
        start_state_prob=np.array([1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0])
        self.start_state=np.random.choice(range(len(start_state_prob)),p=start_state_prob)
        self.current_state=copy.deepcopy(self.start_state)
        self.name="4x4 Maze"
        self.discount=0.95
        self.reward=0
        self.done=False
        
    def step(self,action):
        
        if action==0:
            
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
            
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            self.done=next_state==self.current_state
            
        if action==1:
            
            transition_probability=np.array([[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
        
           
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            self.done=next_state==self.current_state
        
        
        if action==2:
            
            transition_probability=np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
        
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            self.done=next_state==self.current_state
           
        if action==3:
            
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
        
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            self.done=next_state==self.current_state
           
        
        
        
        if next_state==15:
            observation=1
            self.reward=1
        else:
            observation=0
            self.reward=0
        self.current_state=next_state
        return observation,self.reward,self.done
    def number_of_actions(self):
        actions=4
        return actions
    def number_of_observations(self):
        observations=2
        return observations
    def reset(self):
        self.current_state=copy.deepcopy(self.start_state)
        return self.current_state
