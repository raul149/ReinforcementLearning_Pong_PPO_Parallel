import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from collections import deque
import sys
from PIL import Image
import PIL
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import cv2
from skimage import transform
from skimage.color import rgb2gray  # grayscale image


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_space = 0
        self.action_space = 3
        self.hidden = 512
        #self.fc1 = torch.nn.Linear(state_space, self.hidden)  # CNN is now first layer
        self.fc2_mean = torch.nn.Linear(self.hidden, 128)  # neural network for Q (actor) (action-value)
        self.fc2_2 = torch.nn.Linear(128, 3)
        self.fc3 = torch.nn.Linear(self.hidden, 128)  # neural network for V (critic) (state-value)
        self.fc3_2 = torch.nn.Linear(128, 1)
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # Implement learned variance during gradient update of NN's
        self.init_weights()
        # create Convolutional Neural Network: we input 4 frames of dimension of 80x80
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 10, stride=5),  # (number of layers, number of filters, kernel_size e.g. 8x8, stride)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part: Convolutional Neural Network
        #print(x[0])
        x = self.cnn(x)
        #print(x)
        # Actor part
        fc2_1=self.fc2_mean(x)
        fc2_1 = F.relu(fc2_1)
        action_mean = self.fc2_2(fc2_1)
        #If we want to use probs
        #action_mean = F.softmax(action_mean)
        sigma = self.sigma


        # Critic part
        state_val_1 = self.fc3(x)
        state_val_1 = F.relu(state_val_1)
        state_val = self.fc3_2(state_val_1)
        # Instantiate and return a normal distribution with mean mu and std of sigma
        #action_dist = Normal(action_mean, sigma)
        #action_dist = Categorical(logits=action_mean)
        #If we use probs instead of logits
        action_dist = Categorical(logits=action_mean)

        # Return state value in addition to the distribution
        return action_dist, state_val


class Agent(object):
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #Usually for parallel training we run on CPU even if CUDE is available cause has less memory cap.
        #Recommended use of CPU
        device = 'cpu'
        self.train_device = device
        self.policy = Policy().to(self.train_device)
        #self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=2e-4)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-4, betas=(0.9,0.999))
        self.gamma = 0.99
        self.eps_clip = 0.20
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []
        self.actions = []
        self.name = "BeschdePong"
        self.number_stacked_imgs = 4  # we stack up to for imgs to get information of motion

        self.img_collection = [[] for _ in range(30)] #Here in range(X), is the number of processes running at the same time
        self.img_collection_update = [[] for _ in range(30)] #Here in range(X), is the number of processes running at the same time
        self.epochs = 10  # number of epochs for minibatch update

    def clipped_surrogate(self, old_action_probs, new_action_probs, advantage):
        """ Clipped surrogate of PPO paper to make sure that that the updates of the policy are not too big
        params:

        return:
            loss: PPO loss
        """
        # Calculate ratio of new and old action_probs, using exponential, as we use log probs
        ratio = torch.exp(new_action_probs - old_action_probs)
        # Clamp ratio, to the desired ratio, 0,2
        clip = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip)
        # Clipped surrogate is minima
        clipped_surrogate = torch.min(ratio*advantage, clip*advantage)

        # Calculate the estimation by taking the mean of all three parts
        loss_ppo = torch.mean(clipped_surrogate)

        return loss_ppo

    def update_policy(self, episode_number, episode_done=False):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        actions = torch.Tensor(self.actions).to(self.train_device)
        # treat states and next_states differently since they still contain raw images
        states_raw = self.states
        next_states_raw = self.next_states

        # Clear state transition buffers
        self.states, self.next_states, self.action_probs, self.rewards, self.done, self.actions = [], [], [], [], [], []

        # convert raw images to stacked images
        states = []
        next_states = []
        p=0
        #Loop through the whole transition buffer of raw images of (raw frame)state and (raw frame)next state:
        #Build the stack of 4 frames, according to it.
        while p<(len(states_raw)):
            for h in range (30):
                state_stacked = self.stack_images(states_raw[p], h, update=True, nextstate=False)
                states.append( torch.from_numpy(state_stacked).float() )
                next_state_stacked = self.stack_images(next_states_raw[p], h, update=True, nextstate=True)
                next_states.append( torch.from_numpy(next_state_stacked).float() )
                if done[p] == 1:  # important to handle episode endings, so that for the next one it has to fill with new images
                    self.img_collection_update[h] = []
                p=p+1

        states = torch.stack(states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(next_states, dim=0).to(self.train_device).squeeze(-1)

        # Bring states in right order to be processed
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)


        for i in range(self.epochs):
            # get minibatches
            number_transitions = len(states)  # check how many transitions have been collected
            number_batches = int(number_transitions*0.35)  # get mini-batch size
            indices_minibatch = random.sample(range(number_transitions), number_batches)  # randomly sample inidices for minibatch

            # Create mini-batches:
            old_states = torch.stack([states[i] for i in indices_minibatch], dim=0).to(self.train_device).squeeze(-1)
            old_next_states = torch.stack([next_states[i] for i in indices_minibatch], dim=0).to(self.train_device).squeeze(-1)
            old_action_probs = torch.stack([action_probs[i] for i in indices_minibatch], dim=0).to(self.train_device).squeeze(-1)
            old_rewards = torch.stack([rewards[i] for i in indices_minibatch], dim=0).to(self.train_device).squeeze(-1)
            old_done = torch.stack([done[i] for i in indices_minibatch], dim=0).to(self.train_device).squeeze(-1)
            old_actions = torch.stack([actions[i] for i in indices_minibatch], dim=0).to(self.train_device).squeeze(-1)

            # get new state values and action distributions
            action_distributions, pred_states_value = self.policy.forward(old_states)
            action_distributions_next, pred_next_states_value = self.policy.forward(old_next_states)

            # calculate new action probabilities
            new_action_probs = action_distributions.log_prob(old_actions) # Calculate the log probability of the action

            # Delete 1 dimensionality
            pred_next_states_value = (pred_next_states_value).squeeze(-1)
            pred_states_value = (pred_states_value).squeeze(-1)

            # Handle terminal states
            pred_next_states_value = torch.mul(pred_next_states_value, 1-old_done)

            #Critic Loss:
            critic_loss = F.mse_loss(pred_states_value, old_rewards+self.gamma*pred_next_states_value.detach())


            # Compute advantage estimates:
            advantage = old_rewards + self.gamma * pred_next_states_value - pred_states_value


            # calculate PPO loss: We detach Old probs and Advantages
            loss_ppo = self.clipped_surrogate(old_action_probs.detach(), new_action_probs, advantage.detach())
            #Calculate Entropy loss, with coefficient c=0.01, recommended by paper
            entropy_loss = 0.01*action_distributions.entropy()
            entropy_loss=torch.mean(entropy_loss)
            # Loss actor critic: Compute the gradients of loss w.r.t. network parameters
            #Multiplying the critic loss by 0,4. Several numbers such as 0,5 and 0.3 were tried, but 0,4 delivered better results
            loss = 0.5*critic_loss - loss_ppo - entropy_loss
            #print(loss)
            loss.backward()

            # Update network parameters using self.optimizer and zero gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

    def preprocessing(self, observation):
        """ Preprocess the received information: 1) Grayscaling 2) Reducing quality (resizing)
        Params:
            observation: image of pong
        """
        # Grayscaling

        img_gray = np.dot(observation, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        img_resized = cv2.resize(img_gray, dsize=(100, 100))
        #print('First',img_resized)
        img_resized[img_resized < 25] = 0 #Background in black
        img_resized[img_resized >= 25] = 255 #The rest in white
        #print('After',img_resized)
        # Normalize pixel values
        img_norm = img_resized / 255.0
        #print(img_norm.shape)
        # Downsampling: we receive squared image (e.g. 200x200) and downsample by x2.5 to (80x80)

        return img_resized

    def stack_images(self, observation, p, update=False, nextstate=False):
        """ Stack up to four frames together
        Params:
            observation: raw 200x200 image
            update: in case we are updating (True) we need to access different variable self.img_collection_update
        """
        # image preprocessing
        img_preprocessed = self.preprocessing(observation)
        if update == False:
            if (len(self.img_collection[p]) == 0):  # start of new episode, use len() instead of timestep to stay Markovian
                # img_collection get filled with zeros
                self.img_collection[p] = deque([np.zeros((100,100), dtype=np.int) for i in range(4)], maxlen=4)
                # fill img_collection 4x with the first frame
                self.img_collection[p].append(img_preprocessed)
                self.img_collection[p].append(img_preprocessed)
                self.img_collection[p].append(img_preprocessed)
                self.img_collection[p].append(img_preprocessed)
                # Stack the images in img_collection
                img_stacked = np.stack(self.img_collection[p], axis=2)
            else:
                # Delete first/oldest entry and append new image
                self.img_collection[p].append(img_preprocessed)

                #CHECK TO SEE IF THE PICTURES LOOK GOOD (DONE:WORK)
                #np_array = np.array(self.img_collection[p][0])
                #plt.imsave( "Image0_%s_%d.png" % (p, self.timestepss), np_array, cmap='Greys')
                #np_array = np.array(self.img_collection[p][1])
                #plt.imsave( "Image1_%s_%d.png" % (p, self.timestepss), np_array, cmap='Greys')
                #np_array = np.array(self.img_collection[p][2])
                #plt.imsave( "Image2_%s_%d.png" % (p, self.timestepss), np_array, cmap='Greys')
                #np_array = np.array(self.img_collection[p][3])
                #plt.imsave( "Image3_%s_%d.png" % (p, self.timestepss), np_array, cmap='Greys')
                img_stacked = np.stack(self.img_collection[p], axis=2)
                #print(img_stacked.shape)
            return img_stacked
        else:
            if nextstate==True:
                #print('Next State')
                #print(len(self.img_collection_update[p]))
                img_nextstate = self.img_collection_update[p].copy()
                #CHECK TO SEE IF THE PICTURES LOOK GOOD (DONE:WORK)
                """ np_array = np.array(self.img_collection_update[p][0])
                plt.imsave( "Image%s_%d_0.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(self.img_collection_update[p][1])
                plt.imsave( "Image%s_%d_1.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(self.img_collection_update[p][2])
                plt.imsave( "Image%s_%d_2.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(self.img_collection_update[p][3])
                plt.imsave( "Image%s_%d_3.png" % (p, self.timestepss), np_array, cmap='Greys') """
                #Appending the new image
                img_nextstate.append(img_preprocessed)
                #CHECK TO SEE IF THE PICTURES LOOK GOOD (DONE:WORK)
                """ np_array = np.array(img_nextstate[0])
                plt.imsave( "Image%s_%d_40.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(img_nextstate[1])
                plt.imsave( "Image%s_%d_41.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(img_nextstate[2])
                plt.imsave( "Image%s_%d_42.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(img_nextstate[3])
                plt.imsave( "Image%s_%d_43.png" % (p, self.timestepss), np_array, cmap='Greys') """
                #if (self.img_collection_update[p][3] != img_nextstate[3]).all:
                #    print('true')
                # Stack the images in img_collection
                img_stacked = np.stack(img_nextstate, axis=2)
                return img_stacked

            if (len(self.img_collection_update[p]) == 0):  # start of new episode, use len() instead of timestep to stay Markovian
                # img_collection get filled with zeros
                #print('New', p)
                self.img_collection_update[p] = deque([np.zeros((100,100), dtype=np.int) for i in range(4)], maxlen=4)
                # fill img_collection 4x with the first frame
                self.img_collection_update[p].append(img_preprocessed)
                self.img_collection_update[p].append(img_preprocessed)
                self.img_collection_update[p].append(img_preprocessed)
                self.img_collection_update[p].append(img_preprocessed)

                # Stack the images in img_collection
                img_stacked = np.stack(self.img_collection_update[p], axis=2)
            else:
                # Delete first/oldest entry and append new image
                self.img_collection_update[p].append(img_preprocessed)
                #DEBUGGGG CHECKING (WORKS)
                #print(len(self.img_collection_update[p]))
                #print('Adding images')
                #np_array = np.array(self.img_collection_update[p][0])
                """ plt.imsave( "Image_%s_%d_0.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(self.img_collection_update[p][1])
                plt.imsave( "Image_%s_%d_1.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(self.img_collection_update[p][2])
                plt.imsave( "Image_%s_%d_2.png" % (p, self.timestepss), np_array, cmap='Greys')
                np_array = np.array(self.img_collection_update[p][3])
                plt.imsave( "Image_%s_%d_3.png" % (p, self.timestepss), np_array, cmap='Greys') """
                # Stack the images in img_collection
                img_stacked = np.stack(self.img_collection_update[p], axis=2)
            return img_stacked


    def get_action(self, observation, evaluation=False):
        #Loop through the number of processes
        #Create a list of list with 30 lists.
        stacked_img=[[] for _ in range(30)]
        #For every list insert the stack the four pictures of the state
        for p in range(30):
            stacked_img[p] = self.stack_images(observation[p], p)
        #print(stacked_img)
        #Put the stacks as np.array
        stacked_img = np.array(stacked_img)
        #print(stacked_img.shape)
        # create torch out from numpy array
        x = torch.from_numpy(stacked_img).float().to(self.train_device)
        #print(x.shape)
        #Add one more dimension, batch_size=1, for the conv2d to read it

        # Change the order, so that the channels are at the beginning is expected: (1*4*80*80) = (batch size, number of channels, height, width)
        x = x.permute(0, 3, 1, 2)

        # Pass state x through the policy network
        action_distribution, __ = self.policy.forward(x)

        # Get action: Return mean if evaluation, else sample from the distribution (returned by the policy)
        if evaluation:
            action = action_distribution.mean()
        else:
            action = action_distribution.sample()

        # Calculate the log probability of the action
        act_log_prob = action_distribution.log_prob(action)
        action = action.detach().cpu().numpy()

        return action, act_log_prob


    def store_outcome(self, state, next_state, action, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        self.actions.append(action)

    def load_model(self):
        """ Load the model
        """
        weights = torch.load("Model_Scratchdumbestai_WimblepongVisualSimpleAI-v0_42000.mdl", map_location='cpu')
        self.policy.load_state_dict(weights, strict=False)


    def get_name(self):
        """ Get the name of the agent
        """
        return self.name

    def reset(self, i):
        """ Resets the image collection for the environment which has finished
        """
        self.img_collection[i] = []
