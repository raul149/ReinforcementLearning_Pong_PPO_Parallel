"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=70)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale*2
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
import agent_new
import agent_opponent
import agent
player = agent.Agent()
opponent = agent.Agent()
player.load_model()
opponent.load_model2()

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
for i in range(0,episodes):
    done = False
    ob1, ob2 = env.reset()
    timesteps=0
    while not done:
        # Get the actions from both SimpleAIs
        action1 = player.get_action(ob1)
        action2 = opponent.get_action(ob2)
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        timesteps += 1
        #print('Timesteps:',timesteps)
        # Count the wins
        args.headless=False
        if rew1 == 10:
            win1 += 1

        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            print(timesteps)