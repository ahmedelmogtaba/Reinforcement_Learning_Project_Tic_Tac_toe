from copy import deepcopy
import random

from .MCTS import MCTS

class XO_Board() :

  def __init__(self,board = None):
    self.x = 1
    self.o = -1
    self.empty = 0
    self.first_player = 'x'
    self.second_player = 'o'
    self.empty_place = '-'
    self.mcts_randomness = 0.9
    self.user_input = None
    self.WIN_REWARD = 1
    self.DRAW_REWARD = 0
    self.LOSS_REWARD = -1
    self.STUPID_ACTION_REWARD = 0
    self.SMART_ACTION_REWARD = 0
    self.EPS_START = 1.0
    self.mcts_eps_threshold=1.0
    self.EPS_END = 0.9
    self.EPS_DECAY = 8000
    self.agent_winner=0
    self.mcts_winner=0
    self.draw=0
    self.position = {}
    n_actions=9
    self.mcts = MCTS()

    # init the board
    self.init_board()

    # create a copy of previous board state 
    if board:
      self.__dict__ = deepcopy(board.__dict__)
    else:
      self.init_board()

  # initialize the board
  def init_board(self, empty=True):
    for row in range(3):
      for col in range(3):
        # set every square to empty square
        self.position[row,col] = self.empty_place  
    if not empty and  random.random()>0.5:
      r_action = random.randrange(9)
      row, col = self.action_to_ids(r_action+1)
      self.position[row,col] = 'o'


  # get whether the game is draw
  def is_draw(self):
    for row , col in self.position:
      # empty square is available
      if self.position[row,col] == self.empty_place:
        return False
    return True

  # get whether the game is win
  def is_win(self):
    # vertical sequence detection
    for col in range(3):
      # define the winning sequence list
      winning_sequence = []
      for row in range(3):
        if self.position[row , col] == self.second_player:
          # update winning sequence
          winning_sequence.append((row,col)) 
        # if we have 3 elements in the row 
        if len(winning_sequence) == 3 :
          return True

    # horizontal sequence detection
    for row in range(3):
      # define the winning sequence list
      winning_sequence = []
      for col in range(3):
        # if found the same next element in the row
        if self.position[row , col] == self.second_player :
          # update winning sequence
          winning_sequence.append((row,col)) 
        # if we have 3 elements in the row 
        if len(winning_sequence) == 3 :
          return True

    # 1st diagonal sequence detection
    # define the winning sequence list
    winning_sequence = []
    for row in range(3):
      # init column
      col = row
      if self.position[row , col] == self.second_player :
        # update winning sequence
        winning_sequence.append((row,col)) 
      # if we have 3 elements in the row 
      if len(winning_sequence) == 3 :
        return True

    # 2nd diagonal sequence detection  
    # define the winning sequence list
    winning_sequence = []
    # loop over board rows
    for row in range(3):
      # init column
      col = 3-row-1
      if self.position[row , col] == self.second_player :
        # update winning sequence
        winning_sequence.append((row,col)) 
      # if we have 3 elements in the row 
      if len(winning_sequence) == 3 :
        return True
    return False


  # generate legal moves
  def generate_moves(self):
    boards = [] #list of objects
    for row in range(3):
      for col in range(3):
        # make sure the current position is empty
        if self.position[row,col] == self.empty_place :
          # append available actions/board state to action list
          boards.append(self.make_move(row,col))
    # return the list of available actions (board class instances)
    return boards

  # implement make move
  def make_move(self,row,col):
    #create new board instance that inherits from the current state
    board = XO_Board(self)
    #make move 
    board.position[row,col] = self.first_player
    #swap players
    (board.first_player , board.second_player) = (board.second_player , board.first_player)
    return board



  def get_state(self):
    board_str = ''
    for row in range(3):
      for col in range(3):
        board_str += ' %s' % self.position[row,col]
      board_str += '\n'

    print(board_str)

  #Function to play actions
  def step(self, action, mask): #action 1-9
    '''
    action : 1->9
    '''

    reward = 0
    state = self.position_list()

    if (self.is_smart_action(self.position_list(), action-1)):
      reward+= self.SMART_ACTION_REWARD

    if state[action-1]!=0: #action is not available
      print(f'action not avaliable action<0,8> {action-1} , state {state}, mask {mask}')###
      # env.get_state()###
      return state, self.STUPID_ACTION_REWARD, False, state

    ######### agent  #########
    ######### agent  #########

    row, col = self.action_to_ids(action)
    self.__dict__ = deepcopy(self.make_move(row, col).__dict__)

    agent_win = self.is_win()
    if agent_win:
      self.agent_winner+=1
      print('agent won ',self.agent_winner)###
      # self.get_state()###
      reward+=self.WIN_REWARD
      return state, reward, True, None 
    elif self.is_draw():
      self.draw+=1
      print('draw ', self.draw)###
      # self.get_state()###
      reward+=self.DRAW_REWARD
      return state, reward, True, None 
    # print('agent [x]')###
    # self.get_state()###

    ######### mcts  #########
    ######### mcts  #########
    self.mcts_eps_threshold = self.mcts_randomness #self.EPS_END + (self.EPS_START - self.EPS_END) * \
        # math.exp(-1. * steps_done / self.EPS_DECAY)
    # print('mcts eps threshold ',self.mcts_eps_threshold)
    sample = random.random()
    if sample > self.mcts_eps_threshold: #use mcts 1.0->0.9
      best_mcts_move = self.mcts.search(self,800)
      # make AI Agent move ....
      try:
          self.__dict__ = deepcopy(best_mcts_move.board.__dict__)
          
      except:
          raise Exception 

    else:  #choose random action
      mask[action-1] = False #the action played by the agent is no more available
      while True:
        r_action=random.randrange(9)
        if mask[r_action]: #available action
          row, col = self.action_to_ids(r_action+1)
          # print('random')
          self.__dict__ = deepcopy(self.make_move(row, col).__dict__)
          break

    mcts_win = self.is_win()
    if mcts_win:
      self.mcts_winner+=1
      print('mcts won ',self.mcts_winner)###
      # self.get_state()###
      reward+=self.LOSS_REWARD
      return state, reward, True, None 
    elif self.is_draw():
      self.draw+=1
      print('draw ',self.draw)
      # self.get_state()###
      reward+=self.DRAW_REWARD
      return state, reward, True, None 
    # print('mcts [o]')###
    # self.get_state()###
    

    return state, reward, False, self.position_list()   #s, reward, done, new_s


  def action_to_ids(self, action):
    if action==1:
      row=0;col=0
    elif action==2:
      row=0;col=1
    elif action==3:
      row=0;col=2
    elif action==4:
      row=1;col=0
    elif action==5:
      row=1;col=1  
    elif action==6:
      row=1;col=2    
    elif action==7:
      row=2;col=0
    elif action==8:
      row=2;col=1
    elif action==9:
      row=2;col=2
    else:
      raise Exception()
    
    return row, col
  

  def action_sapce(self):
    return [1,2,3,4,5,6,7,8,9]

  def position_list(self):
    state = []
    for i in list(self.position.values()):
      if i=='-':
        state.append(0)
      elif i=='x':
        state.append(1)
      elif i=='o':
        state.append(-1)

    return state

  def is_smart_action(self, position_list, a): #<0,8>
    o=self.o
    if a<3: #0,1,2
        if position_list[:3].count(o)==2:
          return True
    if a>=3 and a<6: #3,4,5
        if position_list[3:6].count(o)==2:
          return True
    if position_list[6:].count(o)==2:
        return True

    if position_list[4]==o:#center
        if a==0 and position_list[8]==o:
          return True
        elif a==2 and position_list[6]==o:
          return True
        elif a==6 and position_list[2]==o:
          return True
        elif a==8 and position_list[2]==o:
          return True
    if a==0 or a==3 or a==6:
        if [position_list[0], position_list[3], position_list[6]].count(o)==2:
          return True
    if a==1 or a==4 or a==7:
        if [position_list[1], position_list[4], position_list[7]].count(o)==2:
          return True
    if a==2 or a==5 or a==8:
        if [position_list[2], position_list[5], position_list[8]].count(o)==2:
          return True
    if a==4:
        if position_list[0]==o and position_list[8]==o:
          return True
        elif position_list[2]==o and position_list[6]==o:
          return True

    return False


    
  # print the board state
  def __str__(self):
    # define board string representation
    board_str = ''
    for row in range(3):
      for col in range(3):
        board_str += ' %s' % self.position[row,col]
      # print new line 
      board_str += '\n'

    # side to move 
    if self.first_player == self.user_input:
      board_str = '\n---------------------------\n "Your Turn ==>  "%s" to go"   \n---------------------------\n\n' %self.user_input + board_str
    else:
      board_str = '\n---------------------------\n "Agent Turn"    \n---------------------------\n\n' + board_str
    return board_str

