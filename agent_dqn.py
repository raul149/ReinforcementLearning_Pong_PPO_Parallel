"""
Based on PyTorch DQN tutorial by Adam Paszke <https://github.com/apaszke>

BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.
"""

import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import cv2
from collections import deque
from PIL import Image
import PIL
from skimage import transform
from skimage.color import rgb2gray  # grayscale image
from utils import Transition, ReplayMemory


class CartpoleDQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(CartpoleDQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class LunarLanderDQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(LunarLanderDQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, frames=4):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 512
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        #self.fc2_mean = torch.nn.Linear(self.hidden, 3)  # neural network for Q
        # create Convolutional Neural Network: we input 4 frames of dimension of 80x80
        self.cnn = nn.Sequential(
            nn.Conv2d(frames, 32, 8, stride=4),  # (number of layers, number of filters, kernel_size e.g. 8x8, stride)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )

    def forward(self, x):
        # Common part
        #x = self.fc1(x)
        #x = F.relu(x)
        #print('shape',x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actiondqn = F.relu(self.fc3(x))

        return actiondqn


class Agent(object):
    def __init__(self, env_name, state_space, n_actions, replay_buffer_size=500000,
                 batch_size=32, hidden_size=64, gamma=0.99):
        self.env_name = env_name
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_device = device
        self.n_actions = n_actions
        self.state_space_dim = state_space
        if "CartPole" in self.env_name:
            self.policy_net = CartpoleDQN(state_space, n_actions, 4)
            self.target_net = CartpoleDQN(state_space, n_actions, 4)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        elif "WimblepongVisualSimpleAI-v0" in self.env_name:
            self.policy_net = Policy(state_space, n_actions, 4)
            self.target_net = Policy(state_space, n_actions, 4)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        else:
            raise ValueError("Wrong environment. An agent has not been specified for %s" % env_name)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8).to(self.train_device)
        non_final_mask = non_final_mask.type(torch.bool)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).to(self.train_device)
        state_batch = torch.stack(batch.state).to(self.train_device)
        action_batch = torch.cat(batch.action).to(self.train_device)
        reward_batch = torch.cat(batch.reward).to(self.train_device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).to(self.train_device)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon=0.05):
        #print('initial get action',state.shape)

        #print('final get action',state.shape)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                #print('a',state)
                state = torch.from_numpy(state)
                #print('b',state)
                state = state.unsqueeze(0)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(3)

    def preprocessing(self, observation):
        """ Preprocess the received information: 1) Grayscaling 2) Reducing quality (resizing)
        Params:
            observation: image of pong
        """
        # Grayscaling
        #img_gray = rgb2gray(observation)
        img_gray = np.dot(observation, [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Normalize pixel values
        img_norm = img_gray / 255.0

        # Downsampling: we receive squared image (e.g. 200x200) and downsample by x2.5 to (80x80)
        img_resized = cv2.resize(img_norm, dsize=(80, 80))
        #img_resized = img_norm[::2.5,::2.5]
        return img_resized

    def stack_images(self, observation, img_collection, timestep):
        """ Stack up to four frames together
        """
        # image preprocessing
        img_preprocessed = self.preprocessing(observation)

        if (timestep == 0):  # start of new episode
            # img_collection get filled with zeros again
            img_collection =  deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)
            # fill img_collection 4x with the first frame
            img_collection.append(img_preprocessed)
            img_collection.append(img_preprocessed)
            img_collection.append(img_preprocessed)
            img_collection.append(img_preprocessed)
            # Stack the images in img_collection
            img_stacked = np.stack(img_collection, axis=2)
        else:
            # Delete first/oldest entry and append new image
            #img_collection.pop(0)
            img_collection.append(img_preprocessed)

            # Stack the images in img_collection
            img_stacked = np.stack(img_collection, axis=2) # TODO: right axis??

        return img_stacked, img_collection

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long().to(self.train_device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.train_device)
        next_state = torch.from_numpy(next_state).float().to(self.train_device)
        state = torch.from_numpy(state).float().to(self.train_device)
        self.memory.push(state, action, next_state, reward, done)

    def load_model(self):
        #load_path = '/home/isaac/codes/autonomous_driving/highway-env/data/2020_09_03/Intersection_egoattention_dqn_ego_attention_1_22:00:25/models'
        #policy.load_state_dict(torch.load("./model50000ep_WimblepongVisualSimpleAI-v0_0.mdl"))
        """ Load already created model
        return:
            none
        """
        weights = torch.load("FROM2100v2WimblepongVisualSimpleAI-v0_1900.mdl", map_location=self.train_device)
        self.policy_net.load_state_dict(weights, strict=False)

    def get_name(self):
        """ Interface function to retrieve the agents name
        """
        return self.name

    def reset(self):

        """ Resets the agent’s state after an episode is finished
        return:
            none
        """
        # TODO: Reset the after one point to the middle




