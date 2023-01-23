import numpy as np
import copy





class CheeseMazeEnv():
    def __init__(self):
        start_state_prob=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0])
        self.start_state=np.random.multinomial(len(start_state_prob),start_state_prob).argmax()
        self.current_state=copy.deepcopy(self.start_state)
        self.name="Cheese Maze"
        self.discount=0.95
        self.reward=0
        self.done=False
       
    def step(self,action):
        if action==0: #actions =0,1,2,3 which stands for N,S,E,W
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            observation_probability=np.array([[1,0,0,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,1,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,0,1,0,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,0,1]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.done=next_state==self.current_state
            
            
        if action==1:
            transition_probability=np.array([[0,0,0,0,0,1,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            
            observation_probability=np.array([[1,0,0,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,1,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,0,1,0,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,0,1]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.done=next_state==self.current_state
            
            
        
        
        if action==2:
            transition_probability=np.array([[0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            
            
            observation_probability=np.array([[1,0,0,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,1,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,0,1,0,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,0,0]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.done=next_state==self.current_state
            
            
            
        if action==3:
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            
            observation_probability=np.array([[1,0,0,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,1,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,0,1,0,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,0,1]])
            next_state=np.random.multinomial(len(transition_probability[self.current_state,:]),transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            self.done=next_state=self.current_state
        
        if next_state==10:
            self.reward=1

        else:
            self.reward=0
        self.current_state=next_state
        
            
            
            
        return observation,self.reward,self.done 
    def number_of_actions(self):
        actions=4
        return actions
    def number_of_observations(self):
        observations=7
        return observations
    def reset(self):
        self.current_state=copy.deepcopy(self.start_state)
        return self.current_state
