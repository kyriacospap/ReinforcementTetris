import numpy as np
import torch as T
from props_Network import PropsDeepQnetwork as Network
from props_buffer import ReplayBuffer

class DQNAgent():
    def __init__(self, dims,epsilon =0.9, gamma = 0.99,lr=0.001,
                 epsilon_greedy_frames = 40_000, epsilon_random_frames =1000,
                 epsilon_min = 0.1, max_memory_length = 100_000,
                 update_network = 100, batch_size =32,num_actions=1,
                 algo='DQNBoard',chkpt_dir='board/dqnTorch',env_name='FullBoard'):

        #initialization of the parameters of the agent
        self.max_memory_length = max_memory_length

        self.epsilon_random_frames = epsilon_random_frames
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_min = epsilon_min
        self.epsilon_greedy_frames = epsilon_greedy_frames
        self.epsilon = epsilon
        self.eps_dec = (epsilon-epsilon_min)/epsilon_greedy_frames
        self.lr=lr
        self.n_actions = num_actions
        self.input_dims=dims
        #self.input_dims=np.array(env.board).shape
        self.frames = 0
        self.learn_step_counter=0
        self.update_network = update_network
        self.algo = algo
        self.chkpt_dir = chkpt_dir
        self.env_name=env_name
        self.memory = ReplayBuffer(max_memory_length, self.input_dims, num_actions)

        self.model = Network(self.lr, self.n_actions,
                            input_dims=self.input_dims,
                            name=self.algo+'_q_model',
                            chkpt_dir=self.chkpt_dir)
        self.target_model = Network(self.lr, self.n_actions,
                            input_dims=self.input_dims,
                            name=self.algo+'_q_targedmodel',
                            chkpt_dir=self.chkpt_dir)

    def choose_action(self, dic):
        #dic=env.get_next_boards()
        pos_states=np.array(list(dic.values()))
        pos_actions=list(dic.keys())
        n=len(pos_states)
        if np.random.random() > self.epsilon:
            # val=[]
            # for s in pos_states:
            #     state = T.reshape(T.tensor([s],dtype=T.float),[1,1,20,10]).to(self.model.device)
            #     q_val = self.model.forward(state)
            #     val.append(q_val)
            # action = pos_actions[np.argmax(val)]
            state=T.reshape(T.tensor(pos_states,dtype=T.float),[n,4]).to(self.model.device)
            q_values=self.model.forward(state)
            action = pos_actions[T.argmax(q_values).item()]
        else:
            action = pos_actions[np.random.choice(len(pos_actions))]
        self.frames +=1
        return action

    def store_transition(self, state, reward, state_, done):
        self.memory.store_transition(state, reward, state_, done)

    def sample_memory(self):
        state, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.model.device)
        rewards=T.tensor(reward).to(self.model.device)
        #rewards = T.reshape(T.tensor(reward),[self.batch_size,1]).to(self.model.device)
        dones = T.tensor(done).to(self.model.device)
        #actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.model.device)

        return states, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.update_network == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self,i):
        self.model.save_checkpoint(i)
        self.target_model.save_checkpoint(i)

    def load_models(self,i):
        self.model.load_checkpoint(i)
        self.target_model.load_checkpoint(i)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.model.optimizer.zero_grad()

        self.replace_target_network()

        states, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        q_pred = self.model.forward(states)[indices,0]
        q_next = self.target_model.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        loss = self.model.loss(q_target, q_pred).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
