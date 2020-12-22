import torch
import gym
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from agent_ac import Agent, Policy
from wimblepong import Wimblepong  # import wimblepong-environment
import pandas as pd
from PIL import Image
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter



# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make(env_name)
    best=0

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape
    observation_space_dim = env.observation_space.shape
    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)
    agent.load_model()

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    counter = 0

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()
        observation = np.array(observation)

        # Loop until the episode is over
        while not done:
            # Get action from the agent, an action gets chosen based on the img_stacked processed.
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation
            env.render()

            # Perform the action on the environment, get new state and reward. Now we perform a new step, to see what happens with the action in our current state, the result is the next state
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            #env.render()
            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, observation, action_probabilities, reward, done)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            counter += 1
            """ if counter%160==0 and episode_number<100001:
                agent.update_policy(episode_number, episode_done=done)
            if counter%320==0 and episode_number<600950 and episode_number>100000:
                agent.update_policy(episode_number, episode_done=done)
            if counter%450==0 and episode_number>600949:
                agent.update_policy(episode_number, episode_done=done) """

            # Update the actor-critic code to perform TD(0) updates every 50 timesteps
            #if timesteps%45==0 and not done:
            #    agent.update_policy(episode_number, episode_done=done)

        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        #COMMENT FOR TASK 2-3 NEXT LINE - UNCOMMENT TASK 1
        #agent.update_policy(episode_number, episode_done=done)
        if avg>best and episode_number >1000:
                best=avg
                torch.save(agent.policy_net.state_dict(),"ACBEST.mdl")
        # save model:
        if episode_number % 6503 == 0 and episode_number != 0:
            torch.save(agent.policy.state_dict(), "ACCOOL2%s_%d.mdl" % (env_name, episode_number))

    # Training is finished - plot rewards
    if print_things:
        plt.plot(timestep_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("AC reward history (non-episodic)")
        plt.show()
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("AC reward history (non-episodic)")
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Non-Episodic AC"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "ACSCRATCH%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=1000000, help="Number of episodes to train for")
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

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        policy.load_state_dict(state_dict)
        #test(args.env, 100, state_dict, args.render_test)
