import math 
import random

# Tree node class 
class MCTS():
  # search for the best move in the current position
  def search(self,initial_state,num_iter):
    # create root node
    self.root = TreeNode(initial_state,None)

    # Do iterations
    for iteration in range(num_iter):
      # select a node (selection phase)
      node = self.select(self.root)

      # score current node (simulation phase)
      score = self.simulate_game(node.board)
      # backpropagate the number of visits and the score
      self.backpropagate(node,score)

    # pick up the best move in the current position
    try:
      return self.get_best_move(self.root,0)
    except:
      pass  

  # select most promising node
  def select(self,node):
    # make sure that it is not a terminal node
    while not node.is_terminal:
      # case where the node is fully expanded
      if node.is_fully_expanded:
        node = self.get_best_move(node,2)
      # case where the node is not fully expanded
      else:
        # expand the nodes
        return self.expand(node)
    return node


  # expand node
  def expand(self,node):
    # generate legal moves for the given node
    states = node.board.generate_moves()
    # loop over generated 
    for state in states:
      # make sure that the current node is not present in child nodes 
      if str(state.position) not in node.children:
        # create new node
        new_node = TreeNode(state,node)
        # add child node to parent's node children
        node.children[str(state.position)] = new_node

        # check if node is fully expanded
        if len(states) == len(node.children):
          node.is_fully_expanded = True
        # return newly created node
        return new_node  


  # simulate the game by making random moves until reach the end of the game
  def simulate_game(self,board):
    # make random moves for both sides until terminal state is reached
    while not board.is_win():
      # try to make a move
      try:
        # make the move on board
        board = random.choice(board.generate_moves())
        
      except:
        # return a draw score    
        return 0
        
    # return the score from player x perspective
    if board.second_player == 'x':
      return 1
    elif board.second_player == 'o':
      return -1    


  #  backpropagate   
  def backpropagate(self,node,score):
    # update node visit count and score up to root node
    while node is not None:
      # update node's visits
      node.visits += 1
      # update node's score
      node.score += score
      # set node to parent
      node = node.parent

  # select the best node based on UCB1 formula
  def get_best_move(self,node,exploration_factor):
    # define best score and best moves
    best_score = float('-inf')
    best_moves = []
    # loop over node's children
    for child in node.children.values():
      # define current player
      if child.board.second_player == 'x': 
        current_player = 1
      elif child.board.second_player == 'o':   
        current_player = -1

      # use UCB1 formula to get the move score
      move_score = current_player * child.score / child.visits + exploration_factor * math.sqrt(math.log(node.visits/child.visits))

      #better move has been found 
      if move_score > best_score:
        best_score = move_score
        best_moves = [child]

      # move score is equal to the best score  
      elif move_score == best_score :
        best_moves.append(child)

    # return one of the best moves randomly
    return random.choice(best_moves)

class TreeNode():
  def __init__(self,board,parent):
    self.board = board

    # check if the node is terminal
    if self.board.is_win() or self.board.is_draw():
      # that means the game is over
      self.is_terminal = True 
    else :
      # we have a non terminal node
      self.is_terminal = False  

    # initialise is fully expanded flag
    self.is_fully_expanded = self.is_terminal
    # initialise parent node if available 
    self.parent = parent  
    # initialize the number of node visits
    self.visits = 0

    # initialize the total score of the node
    self.score =  0

    # initialize the current node children
    self.children = {}