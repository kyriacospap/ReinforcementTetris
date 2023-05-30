import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class PropsDeepQnetwork(nn.Module):
    def __init__(self,lr,n_actions,name,input_dims,chkpt_dir):
        super(PropsDeepQnetwork,self).__init__()
        self.checkpoint_dir=chkpt_dir
        self.tempdir=os.path.join(self.checkpoint_dir,'models')
        self.checkpoint_file=os.path.join(self.tempdir,name)

        # self.conv1=nn.Conv2d(input_dims[0],32,3,stride=4)
        # self.conv2=nn.Conv2d(32,64,2,stride=2)
        # self.conv3=nn.Conv2d(64,64,1,stride=1)
        # self.conv1=nn.Conv2d(input_dims,32,3)
        # self.conv2=nn.Conv2d(32,64,3)
        # self.conv3=nn.Conv2d(64,64,3)

        #fc_input_dims=self.calculate_conv_output_dims(input_dims)
        self.fc1=nn.Linear(5,32)
        self.fc2=nn.Linear(32,64)
        self.fc3=nn.Linear(64,n_actions)

        #self.optimizer=optim.RMSprop(self.parameters(),lr=lr)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)

        self.loss=nn.MSELoss()
        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self,input_dims):
        state=T.zeros(1,input_dims)
        dims=self.conv1(state)
        dims=self.conv2(dims)
        dims=self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self,state):
        # conv1=F.relu(self.conv1(state))
        # conv2=F.relu(self.conv2(conv1))
        # conv3=F.relu(self.conv3(conv2))
        # # conv3 shape is BatchSize x n_filters x H x W
        # conv_state=conv3.view(conv3.size(0),-1)
        flat1=F.relu(self.fc1(state))
        flat2=F.relu(self.fc2(flat1))
        actions=self.fc3(flat2)

        return actions

    def save_checkpoint(self,i):
        print('...saving checkpoint...',flush=True)
        T.save(self.state_dict(),self.checkpoint_file+str(i))

    def load_checkpoint(self,i):
        print('...loading checkpoint...',flush=True)
        self.load_state_dict(T.load(self.checkpoint_file))
