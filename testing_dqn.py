import gym
from gym.wrappers import TimeLimit, Monitor
import numpy as np
from matplotlib import pyplot as plt
from agent_dqn import Agent as DQNAgent  # Task 4
from itertools import count
import torch
import cv2
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils import plot_rewards
import seaborn as sb
from wimblepong import Wimblepong # import wimblepong environment
import pandas as pd
from PIL import Image
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0", help="Environment to use")
parser.add_argument("--train_episodes", type=int, default=200, help="Number of episodes to train for")
parser.add_argument("--render_test", action='store_true', help="Render test")
parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                    help='run the script in production mode and use wandb to log outputs')
parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                    help='weather to capture videos of the agent performances (check out `videos` folder)')
parser.add_argument('--wandb-project-name', type=str, default="highway-env",
                    help="the wandb's project name")
parser.add_argument('--wandb-entity', type=str, default=None,
                    help="the entity (team) of wandb's project")
args = parser.parse_args()



env_name = "WimblepongVisualSimpleAI-v0"
env = gym.make(env_name)
env.reset()

 #Values for DQN  (Task 4)
if "CartPole" in env_name:
    TARGET_UPDATE = 50
    glie_a = 500
    num_episodes = 2000
    hidden = 12
    gamma = 0.95
    replay_buffer_size = 500000
    batch_size = 256
elif "WimblepongVisualSimpleAI-v0" in env_name:
    TARGET_UPDATE = 17
    eps = 0.25
    num_episodes = 35000
    hidden = 64
    gamma = 0.99
    replay_buffer_size = 45000
    batch_size = 128
else:
    raise ValueError("Please provide hyperparameters for %s" % env_name)

# The output will be written to your folder ./runs/CURRENT_DATETIME_HOSTNAME,
# Where # is the consecutive number the script was run
""" exp_name = 'SCRATCH'
experiment_name = exp_name
data_path = os.path.join('data', experiment_name)
models_path = f"{data_path}/models"
import wandb
wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=exp_name, monitor_gym=True, save_code=True)
writer = SummaryWriter(f"/tmp/{exp_name}") """

action_space_dim = env.action_space.shape
observation_space_dim = env.observation_space.shape


# Task 4 - DQN
agent = DQNAgent(env_name, observation_space_dim, action_space_dim, replay_buffer_size, batch_size, hidden, gamma)
agent.load_model()
# Training loop
cumulative_rewards = []
timestep_history = []
average_reward_history = []
average_timestep_history = []

totaltimesteps = 0
for ep in range(num_episodes):
    # Initialize the environment and state
    print(ep)
    timesteps=0
    observation = env.reset()
    done = False
    observation = np.array(observation)
    img_collection =  deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)
    #We send the first one, will the full of zeros, and the initial observation which is our 'state'.
    state_images, img_collection = agent.stack_images(observation,img_collection, timestep=timesteps)
    eps = eps*0.99975
    if eps<0.10:
        eps=0.10
    cum_reward = 0
    eps=0
    while not done:
        # Select and perform an action
        action = agent.get_action(state_images, eps)
        next_observation, reward, done, _ = env.step(action)
        cum_reward += reward
        if done:
            next_observation = next_observation*0
        # We process the images to get the proper stat rather than just the observation.
        next_state_images, img_collection = agent.stack_images(next_observation,img_collection, timestep=timesteps)
        timesteps += 1
        totaltimesteps += 1
        env.render()
        # Task 1: TODO: Update the Q-values
        #agent.single_update(state, action, next_state, reward, done)
        # Task 2: TODO: Store transition and batch-update Q-values
        #agent.store_transition(state_images,action,next_state_images,reward,done)
        #agent.update_estimator()
        # Task 4: Update the DQN
        #agent.update_network()
        # Move to the next state
        state_images = next_state_images
    cumulative_rewards.append(cum_reward)
    """writer.add_scalar('Training ' + env_name, cum_reward, ep)
    writer.add_scalar('Training Timesteps ' + env_name, timesteps, ep)"""
    print('Timesteps:',timesteps,'Reward:', cum_reward)
    #Store data
    timestep_history.append(timesteps)
    if ep > 100:
        avg = np.mean(cumulative_rewards[-100:])
        avg2 = np.mean(timestep_history[-100:])
    else:
        avg = np.mean(cumulative_rewards)
        avg2 = np.mean(timestep_history)
        average_reward_history.append(avg)
        average_timestep_history.append(avg2)
        avg3 = np.mean(average_reward_history)
        avg4 = np.mean(average_timestep_history)
    print('Total Avg Timesteps:',avg4,'Reward:', avg3)
    print('Avg . 100 . Timesteps:',avg2,'Reward:', avg)



    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4

    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()
#
#    # Save the policy
#    # Uncomment for Task 4
    if ep % 450 == 0:
        torch.save(agent.policy_net.state_dict(),"INCREASING_DIF_AI%s_%d.mdl" % (env_name, ep))

plot_rewards(cumulative_rewards)
print('Complete')
plt.ioff()
plt.show()



