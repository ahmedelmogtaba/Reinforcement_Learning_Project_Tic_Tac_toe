# Reinforcement_Learning_Project_Tic_Tac_toe
Final Project for Reinforcement Learning (AMMI)



![Screenshot from 2021-09-12 21-46-46](https://user-images.githubusercontent.com/45710249/134741268-83b36ffc-d690-430f-941b-2912ea76b0c1.png)


In this project, we used deep reinforcement learning to train an agent to play tic tac toe game.We first built an environment for the tic-tac-toe game. Then we implemented an MCTS agent. Finally we utilized the MCTS agent to train a DQN to play the tic tac toe game.

our work :

1. Implementation of Monte-Carlo Tree Search (MCTS) to play tic tac toe game.
2. Tic tac toe game environment.
3. Implementation of deep-Q learning algorithm to train RL agent to play tic tac toe game.

# MONTE CARLO TREE SEARCH

Monte-Carlo Tree search is made up of four distinct operations:
1. Tree Traversal/Selection.
2. Node Expansion.
3. Rollout (random simulation).
4. Backpropagation

![monti-carlo-flow-chart](https://user-images.githubusercontent.com/45710249/134753609-f9befd92-5ab1-479a-a15b-16c43984f9b8.jpeg)

# DQN
- Initialize replay memory capacity
- Initialize the network with random weights
- Clone the policy network, and call it the target network
- For each episode:
1. Initialize the starting state
2. For each time step:
    i. Select an action: via exploration or exploitation
    ii. Execute selected action in a emulator
    iii. Observe reward and next state
    iv. Store experience in replay memory Sample random batch from replay memory
    v. Preprocess states from batch
    vi. Pass batch of preprocessed states to policy network
    vii. Calculate loss between output Q values and target Q values:
    • Require a pass to the target network for the next state
    viii. Gradient descent updates weights in the policy network to minimize loss:
    • After x time steps, weights in the target network are updated to the weights in
    the policy network.


