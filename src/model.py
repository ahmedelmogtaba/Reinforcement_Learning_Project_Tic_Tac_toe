import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



def select_action(state, policy_net, EPS_END, EPS_START , EPS_DECAY, steps_done, mask=None): #eps-greedy 
    '''
    return action <0,8>
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_actions=9
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # print('select action, agent',eps_threshold)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if mask is not None:
              # mask = np.array([1,1,1,0,0,0,1,1,1]) #mask =1 where empty
              ids = torch.tensor(np.where(mask)).squeeze()
              p = policy_net(state).squeeze().cpu()
              max_value = p[ids].max(0)[0].view(1)
              # print(f'select action agent ids: {ids} p[ids].max(0)[1] {p[ids].max(0)[1]}')
              # return p[ids].max(0)[1].view(1)
              max_index = (p == max_value).nonzero(as_tuple=True)[0][0].view(1)
              # print('action =  ' , max_index)
              return max_index, eps_threshold
            else:
              return policy_net(state).max(1)[1].view(1), eps_threshold

    else:
        # print('select action, random',eps_threshold)
        while True:
          action=random.randrange(n_actions)
          if mask[action]:
            break
        return torch.tensor([action], device=device, dtype=torch.long), eps_threshold

criterion = nn.SmoothL1Loss() #nn.MSELoss() 
def optimize_model(criterion=criterion):

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) #[(state, action, next_state, reward)]


    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool) # s=[2,3,None,4] ==> tensor([ True,  True, False,  True])
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None]).to(device)

    state_batch = torch.stack(batch.state)
    # state_batch = [[0,2,1,1,0,2,1,2,1],
    #                [0,1,2,0,0,0,0,1,0]]
    action_batch = torch.stack(batch.action)
    # action_batch = [1,3]
    reward_batch = torch.stack(batch.reward)
    #reward_batch = [-1,0]
    # print('batch non_final_next_states',non_final_next_states))###
    # print('batch state_batch',state_batch))###
    # print('batch action_batch',action_batch))###
    # print('batch reward_batch',reward_batch))###

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch.to(device))  #Qo(s,a) == state_action_values = [22,14] for action_batch= [1,3]

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[final_state]=0 (the next value for the final state is 0)
    next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach() #max[Qt(s,a)] 
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values.view(256,1))

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # print('state_action_values, expected_state_action_values.unsqueeze(1)'))###
    # print(state_action_values, expected_state_action_values.unsqueeze(1)))###
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print('shape ',state_action_values.shape, expected_state_action_values.shape)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item(), state_action_values.mean().item()

