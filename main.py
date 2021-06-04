import torch 
import matplotlib.pyplot as plt 
from Envs.Env import Env
from Agents.Agents import *


torch.manual_seed(1234)  # python random number generator seed

lr = 0.1  # learning rate
gamma = 0.95  # gamma parameter
num_episodes = 10000  # number of steps (episodes) in epsilon log-space



# create Environment/ Import environment
env = Env()

obs_space_len = [env.Horizon+1]
action_space_len = 2
agent = SingleAgent(lr=lr, gamma=gamma, numEpisodes=num_episodes, epsilon_type='wavy', horizon = 20, action_space = env.action_space)

running_delta = []  # running delta (e.g. the last running_len delta update)
running_acc = []  # running accuracy (e.g. the last running_len accuracy)



for episode_i in range(num_episodes): # episode loop
    done = False
    delta_update = []  # delta update of our Q-table
    n_successes: int = 0  # number of optimal actions (actions with maximum reward)
    cumul_r: float = 0.0  # cumulative reward
    o = env.reset()

    while not done:
        a =  agent.act(o = o,  inumEP = episode_i)
        o_prime, r, done, _ = env.step(action=a, o=o)
        delta_update.append(agent.update(o=o,action=a,r=r,o_prime=o_prime))  # update agent with transition, get delta update
        o = o_prime
        cumul_r += r  # add reward to cumulative reward
        n_successes += int(r > 0.0)  # success if optimal action-reward of 1.0

    running_acc.append(n_successes / env.Horizon)  # add latest accuracy to running data
    running_delta.append(sum(delta_update).item())  # add latest update delta to running data
  # print(running_delta)
  
    print(
    'episode {episode_i}: cumul_reward={cumul_r}, accuracy:{acc:0.1}, '
    'cumul_delta={cumul_delta:0.1}, eps={eps:0.1}'.format(
        episode_i=episode_i, cumul_r=cumul_r, acc=running_acc[-1],
        cumul_delta=running_delta[episode_i], eps=agent.epsilon[episode_i]
    )
    )




env.close()  # close gym environment

plt.plot(running_acc)
plt.show()
