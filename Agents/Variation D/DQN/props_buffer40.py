import numpy as np
from collections import deque
import pickle

class ReplayBuffer_Actions(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, reward, state_, done,action):
        index = self.mem_cntr % self.mem_size
        # if self.mem_cntr< self.mem_size:
        #     self.action_memory.append(action)
        # else:
        #     self.action_memory[index]=action

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        # print(self.action_memory)
        # actions = np.array(self.action_memory)[batch]
        actions = self.action_memory[batch]
        return states, rewards, states_, terminal,actions

    def save_buffer(self,folder):
        print('...saving buffer...',flush=True)
        pickle.dump( self.state_memory, open( folder+'/buffer/state.p', "wb" ))
        pickle.dump( self.new_state_memory, open( folder+'/buffer/next_state.p', "wb" ))
        pickle.dump( self.reward_memory, open( folder+'/buffer/rewards.p', "wb" ))
        pickle.dump( self.terminal_memory, open( folder+'/buffer/dones.p', "wb" ))
        print('...buffer saved succesfully...')

    def load_buffer(self,folder):
        print('...loading buffer...',flush=True)
        self.state_memory=pickle.load(open(folder+'/buffer/state.p','rb'))
        self.new_state_memory=pickle.load(open(folder+'/buffer/next_state.p','rb'))
        self.reward_memory=pickle.load(open(folder+'/buffer/rewards.p','rb'))
        self.terminal_memory=pickle.load(open(folder+'/buffer/dones.p','rb'))
