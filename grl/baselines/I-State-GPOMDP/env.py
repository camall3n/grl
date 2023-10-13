import numpy as np
import copy

class Env():
    def __init__(self, pomdp):
        self.start_state_prob = pomdp.start
        self.start_state=np.random.multinomial(len(self.start_state_prob),self.start_state_prob).argmax()
        # TODO why not this
        # self.start_state = np.random.choice(len(self.start_state_prob), p=self.start_state_prob)
        self.current_state = copy.deepcopy(self.start_state)
        # can't find in info?
        self.name = "POMDP Name"
        self.discount = pomdp.discount
        # ?
        self.reward = 0
        self.done = False

        # changed into parameters
        self.n_actions = len(pomdp.actions)
        self.n_observations = len(pomdp.observations)
        # (nact,nsta,nsta)
        self.rewards = pomdp.R
        # (nact,nsta,nsta)
        self.transitions = pomdp.T
        # (nact,nsta,nobs)
        self.observations = pomdp.Z
        # TODO Pi_phi
    
    def step(self, action):
        for i in range(self.n_actions):
            if action == i:
                transition_probability = self.transitions[i]
                observation_probability = self.observations[i]
                # # debug
                # print("in step, action "+str(i)+" "+str(self.rewards.shape))
                next_state = np.random.multinomial(
                    len(transition_probability[self.current_state,:]),
                    transition_probability[self.current_state,:]
                    ).argmax()
                observation = np.random.multinomial(
                    len(observation_probability[next_state,:]),
                    observation_probability[next_state,:]
                    ).argmax()
                self.reward = self.rewards[i, self.current_state, next_state]
                self.done = next_state==self.current_state
        self.current_state = next_state
        return observation, self.reward, self.done

    def number_of_actions(self):
        return self.n_actions
    
    def number_of_observations(self):
        return self.n_observations

    def reset(self):
        self.current_state = copy.deepcopy(self.start_state)
        return self.current_state
