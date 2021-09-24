from src import XO_Board, MCTS, ReplayMemory, DQN
from src import MEMORY_SIZE,  BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY , TARGET_UPDATE , MEMORY_SIZE 
from src import select_action, optimize_model
import torch
import random
from itertools import count
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = 9

policy_net = DQN(outputs=n_actions, num_embeddings=3, embedding_dim=16).to(device)
target_net = DQN(outputs=n_actions, num_embeddings=3, embedding_dim=16).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters(),lr=0.0001)

env = XO_Board()
mcts = MCTS()
env.init_board()
memory = ReplayMemory(MEMORY_SIZE)

num_episodes = 1000
episode_reward_list=[]
q_values=[]
global steps_done
steps_done=0

losses_list = []
reward_list=[]
q_list=[]
tr_eps=[]
for i_episode in range(num_episodes):
    episode_loss=[]
    episode_q = []
    episode_reward=[]

    print('game #num: ',i_episode+1)
    # Initialize the environment and state
    env.init_board(empty=False)
    state = env.position_list()
    for t in count():
        steps_done+=1

        # Select and perform an action
        mask=[]
        for i in env.position_list():
          mask.append(i==0)

    
        action, eps_threshold = select_action(torch.tensor(state).to(device), policy_net, EPS_END, EPS_START, EPS_DECAY, steps_done, mask=np.array(mask)) #0->8np.array(mask)
        action = action.cpu()
        _, reward, done, new_s = env.step(action.item()+1,mask) #<1,9>
        # print('new s', new_s)
        episode_reward.append(reward)
        
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        if new_s:
          new_s = torch.tensor(new_s)
        
        if not np.all(np.array(torch.tensor(state))== np.array(new_s)):
          memory.push(torch.tensor(state), action, new_s, reward)
        else:
          print('eorrrrrrrrrr in adding the data sample')

        # Move to the next state
        state = new_s

        # Perform one step of the optimization (on the policy network)
        if len(memory) > BATCH_SIZE:
          loss, q = optimize_model() #BSGD
          episode_q.append(q)
          episode_loss.append(loss)


        if done:
            break


    # print('################################################################################################')
    # print('episode_reward : ' , episode_reward)
    # print('################################################################################################')
          

    episode_reward_list.append(episode_reward)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print('game loss: ',np.mean(episode_loss), 'q value',np.mean(episode_q), ' reward :',np.sum(episode_reward), 'exp eps: ',eps_threshold)


print('Complete')
plt.ioff()
plt.plot(q_list)
plt.show()
