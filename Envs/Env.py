# Environemnt Class 
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch 
import gym 
from gym import spaces
from PUs.PU import PU 

class Env(gym.Env): 

    def __init__(self, Horizon=20, num_PU =10, num_SU=5) -> None:
        super(Env,self).__init__()
        
        self.observation_space = spaces.Discrete(Horizon + 1 )
        self.Horizon = Horizon
        self.Timer = 0 
        self.num_agents = num_SU
        self.num_PU = num_PU
        self.createPU(Horizon)
        self.action_space = spaces.Box(low=0, high=2, shape=(self.num_agents, self.num_PU),dtype=int)
        

    def createPU (self, Horizon):
        PUs = [] 
        TxPattern = torch.zeros(Horizon,self.num_PU)
        for i in range(self.num_PU): 
            PU1= PU(Horizon)
            PU1.createTxPattern()
            PUs.append(PU1)
            TxPattern [:, i] = PU1.TxPattern 
        
        self.TxPattern =  TxPattern 



    def reset(self):
        self.Timer = 0
        # o = torch.tensor([self.Timer], dtype=torch.float)
        # o = torch.tensor([self.Timer])
        o = self.Timer
        return o
    

    def step(self, action, o ): #action is a binary  0 or 1, in case of single PU 
        # assert self.action_space.contains(action[0])
        # dimmension 0 
        
        
        r = torch.zeros(self.num_agents,1)
        
        self.Timer = int(o)
        if self.Timer < self.Horizon:  # non-terminal observation, horizon not reached
            # r = torch.sum((action == self.TxPattern[self.Timer,:]).astype(torch.int)) 
            
            for i_agent in range(self.num_agents):
                r[i_agent] = torch.sum((action[i_agent,:] == self.TxPattern[self.Timer,:]).long())


        else:  # gym horizon reached
            for i_agent in range(self.num_agents):
                r[i_agent] = 0.0  
            

        self.Timer += 1  # increment our time-step / observation
        # o = torch.tensor([self.Timer], dtype=torch.float) # observation that will return
        # o = torch.tensor([self.Timer]) # observation that will return
        o = self.Timer # observation that will return
        done = (self.Timer == torch.tensor([self.Horizon]))  # is terminal gym state reached
        # print("o: {} r: {} action: {}".format(o,r,action))
        return o, r, done, {}  # gyms always returns <obs, reward, terminal obs reached, debug/info dictionary>

    def render(self, mode='human'):
        print('Lets see TxPattern \n')
        print(self.TxPattern)


if __name__ == "__main__": 
    TestEnv = Env()
    print(TestEnv.TxPattern)