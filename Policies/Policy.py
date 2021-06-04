
import torch
import  matplotlib.pyplot as plt
# Eplsilon Greedy 

# Wavy Epsilon 

# UCB 

# UCB-H 

import torch
class wavyexponential:
  def __init__(self, numEpisodes:int, sigma = 5):
    self.pi=3.14
    self.expo = torch.exp(-sigma*torch.linspace(0,1,numEpisodes)) 
    self.x = torch.linspace(0, sigma*2*self.pi,numEpisodes)
    self.wave = torch.abs(torch.cos(self.x) )
    self.epsilon = (self.wave+self.expo)*self.expo
    self.epsilon[self.epsilon>1] = 1
    self.expo = torch.exp(-sigma*1.2*torch.linspace(0,1,numEpisodes)) 
    # self.epsilon = self.epsilon/max(self.epsilon)
    self.numEpisodes = numEpisodes

  def render(self):
    x = torch.linspace(0,self.numEpisodes-1,self.numEpisodes)
    stdfigsize=(3.51*1.5,2.2*1.5)
    fig = plt.figure(figsize=stdfigsize) #Width, height 
    plt.plot(x,self.epsilon,'r-',label="wavy-expo")
    plt.plot(x,self.expo,'b-.',label="reg-expo")
    plt.grid(True)
    plt.legend(loc='upper right',shadow=True,fontsize='large')
    plt.xlabel('Episode Number', fontsize='large')
    plt.ylabel('$\epsilon$', fontsize='large')
    plt.show()
 #   plt.savefig(f"{images_dir}/wavyexponential.eps")

# Epsilonfn = wavyexponential(1000,5)
# # print(Epsilonfn.epsilon)
# Epsilonfn.render()

class regexponential:
  def __init__(self, numEpisodes:int, sigma = 5):
    self.expo = torch.exp(-sigma*1.2*torch.linspace(0,1,numEpisodes)) 
    # self.x = torch.linspace(0, sigma*2*pi,numEpisodes)
    # self.wave = torch.abs(torch.cos(x) )
    self.epsilon = self.expo 
    self.numEpisodes = numEpisodes

  def render(self):
    x = torch.linspace(0,self.numEpisodes-1,self.numEpisodes)
    stdfigsize=(3.51*1.5,2.2*1.5)
    fig = plt.figure(figsize=stdfigsize) #Width, height 
    plt.plot(x,self.epsilon,'r-',label="reg-expo")
    # plt.plot(x,self.expo,'b-.',label="Envelope")
    plt.grid(True)
    plt.legend(loc='upper right',shadow=True,fontsize='large')
    plt.xlabel('Episode Number', fontsize='large')
    plt.ylabel('$\epsilon$', fontsize='large')
    plt.show()
#    plt.savefig(f"{images_dir}/regularexponential.eps")
