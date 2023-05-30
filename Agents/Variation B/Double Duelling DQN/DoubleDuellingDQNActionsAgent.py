import numpy as np
import torch as T
from DuellingNetworkActions import DuellingBoardDeepQnetwork as Network
from abuffer import ReplayBuffer_Actions

class DDuellingDQNAgent():
    def __init__(self, dims,all_actions,epsilon =0.9, gamma = 0.99,lr=0.001,
                 epsilon_greedy_frames = 40_000, epsilon_random_frames =1000,
                 epsilon_min = 0.1, max_memory_length = 100_000,
                 update_network = 100, batch_size =32,num_actions=40,
                 algo='DDuellingDQNActions',chkpt_dir='board/dqnTorch',env_name='FullBoard'):

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
        self.memory = ReplayBuffer_Actions(max_memory_length, self.input_dims, num_actions)
        self.action_dic=all_actions
        self.action_dictionary_f={all_actions[i]:i for i in range(len(all_actions))}
        self.action_dictionary_b={i:all_actions[i] for i in range(len(all_actions))}
        self.model = Network(self.lr, self.n_actions,
                            input_dims=self.input_dims,
                            name=self.algo+'_q_model',
                            chkpt_dir=self.chkpt_dir)
        self.target_model = Network(self.lr, self.n_actions,
                            input_dims=self.input_dims,
                            name=self.algo+'_q_targedmodel',
                            chkpt_dir=self.chkpt_dir)
        self.ex_dims=[i for i in dims]
        self.ex_dims.insert(0,1)
        
    def choose_action(self,obs):
        if np.random.random() > self.epsilon:
            state=T.reshape(T.tensor(obs,dtype=T.float),self.ex_dims).to(self.model.device)
            _,adv=self.model.forward(state)
            action = T.argmax(adv).item()
        else:
            action = np.random.choice(self.n_actions)
        self.frames +=1
        return action

    def store_transition(self, state, reward, state_, done,action):
        self.memory.store_transition(state, reward, state_, done,action)

    def sample_memory(self):
        state, reward, new_state, done,action = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.model.device)
        rewards=T.tensor(reward).to(self.model.device)
        #rewards = T.reshape(T.tensor(reward),[self.batch_size,1]).to(self.model.device)
        dones = T.tensor(done).to(self.model.device)
        actions = T.tensor(action).to(self.model.device)
        states_ = T.tensor(new_state).to(self.model.device)

        return states, rewards, states_, dones , actions

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

        states, rewards, states_, dones,actions = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_pred,A_pred = self.model.forward(states)
        V_next,A_next = self.target_model.forward(states_)

        V_eval,A_eval = self.model.forward(states_)

        # q_pred=T.add(V_pred,(A_pred - A_pred.mean(dim=1,keepdim=True)))
        # q_next=T.add(V_next,(A_next - A_next.mean(dim=1,keepdim=True)))
        #
        # # print('INDICES : \n',indices)
        # print('Q_Prediction ::\n')
        # print(q_pred.shape,'\n')
        # print(q_pred,'\n')
        # print('Q_NEXT ::\n')
        # print(q_next.shape,'\n')
        # print(q_next,'\n')

        q_pred=T.add(V_pred,(A_pred - A_pred.mean(dim=1,keepdim=True)))[indices,actions]
        q_next=T.add(V_next,(A_next - A_next.mean(dim=1,keepdim=True)))#.max(dim=1)[0]

        q_eval=T.add(V_eval,(A_eval - A_eval.mean(dim=1,keepdim=True)))

        max_actions=T.argmax(q_eval,dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices,max_actions]

        # print('Q_Prediction ::\n')
        # print(q_pred.shape,'\n')
        # print(q_pred,'\n')
        # print('Q_NEXT ::\n')
        # print(q_next.shape,'\n')
        # print(q_next,'\n')
        # print('Rewards ::\n')
        # print(rewards.shape,'\n')
        # print(rewards,'\n')
        #
        # print('Q_Target ::\n')
        # print(q_target.shape,'\n')
        # print(q_target,'\n')
        #
        # print('Q_Target 2::\n')
        # print(q_target2.shape,'\n')
        # print(q_target2,'\n')
        # q_target=1

        loss = self.model.loss(q_target, q_pred).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
