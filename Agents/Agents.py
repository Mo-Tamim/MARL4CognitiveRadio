import torch 
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SUs.SU import SU
from Policies.Policy import *

class SingleAgent(SU): 
    
    def __init__(self, lr=0.1, gamma=0.1, numEpisodes=200, epsilon_type='regular', horizon = 20, action_space = None):
        super(SingleAgent, self).__init__()
        self.lr = lr #learning rate
        self.gamma = gamma
        self.numEpisodes = numEpisodes
        self.qtable = torch.zeros(horizon+1, 2)  # Q-table with time-step as keys for a 2D Q-table
        self.epsilon_type = epsilon_type
        self.horizon = horizon
        self.action_space = action_space
        self.CreateEpsilonFunction()


    def CreateEpsilonFunction(self,sigma=5):
        if self.epsilon_type == 'wavy': #The default epsilon type is regular-exponential, the other option is 'wavy'
            Epsilonfn = wavyexponential(self.numEpisodes,sigma=sigma)
      # Epsilonfn.render()
        elif self.epsilon_type == 'regular':
            Epsilonfn = regexponential(self.numEpisodes,sigma=sigma)
        elif self.epsilon_type == 'constant':
            Epsilonfn = regexponential(self.numEpisodes,sigma=sigma)
            Epsilonfn.epsilon = torch.ones((self.numEpisodes,1))*0.1
        
        self.epsilon = Epsilonfn.epsilon
        # Epsilonfn.render()
    

    def act(self, inumEP, o ): 
        if torch.rand(1) < self.epsilon [inumEP]: 
            return self.action_space.sample()
        return self.qtable[o].max(0)[1]

    def update(self, o: int, action: int, r: float, o_prime: int):  # agent update function (e.g. Q-learning update)

        old_o_a_value  = self.qtable[o][action]  #store the Q-value for observation o and action a 
        q_prime  = self.qtable[o_prime].max(0)[0] # estimate of optiomal future value (maximum Q-value in observation o_prime)

        # Update Q-table based on the equation above
        td_target = r + self.gamma * q_prime
        self.qtable[o][action] += self.lr* (td_target - old_o_a_value)
        return self.qtable[o][action]-old_o_a_value # return delta update to training loop


if __name__ == "__main__": 
    s = SingleAgent()
    s.CreateEpsilonFunction()
    print(s.epsilon)
   