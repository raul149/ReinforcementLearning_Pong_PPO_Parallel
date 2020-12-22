import torch
import gym
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from agent_ppo import Agent, Policy
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

    exp_name = 'PPO_TESTING_EVFALSE'
    experiment_name = exp_name
    data_path = os.path.join('data', experiment_name)
    models_path = f"{data_path}/models"
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=exp_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{exp_name}")

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape
    observation_space_dim = env.observation_space.shape

    # Instantiate agent and its policy
    agent = Agent()
    agent.load_model() # TODO: uncomment if new model should be created

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    counter = 0
    best=0
    wonp=0
    lost=0

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()
        agent.reset()  # reset all remaining state transition buffers

        # Loop until the episode is over
        while not done:
            # Get action from the agent, an action gets chosen based on the img_stacked processed.

            action, action_probabilities = agent.get_action(observation, evaluation=True)
            previous_observation = observation
            #env.render()

            # Perform the action on the environment, get new state and reward. Now we perform a new step, to see what happens with the action in our current state, the result is the next state
            observation, reward, done, info = env.step(action)

            #env.render() # TODO: uncomment to test and see how it plays pong
            # Store action's outcome (so that the agent can improve its policy)
            #agent.store_outcome(previous_observation, observation, action_probabilities, reward, done, action)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            counter += 1
        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))
            if reward_sum==10:
                wonp=wonp+1
            if reward_sum==-10:
                lost=lost+1
                print("Winning Rate EVFALSE:", wonp/(wonp+lost))
                print("Won:" ,wonp)
                print("Lost:" ,lost)

        # Update the actor-critic code to perform TD(0) updates every 50 timesteps
        #if counter > 500:
        #    counter = 0
        #    agent.update_policy(episode_number, episode_done=done)
        writer.add_scalar('Training Reward' + env_name, reward_sum, episode_number)
        writer.add_scalar('Training Timesteps ' + env_name, timesteps, episode_number)



        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        #if episode_number % 1000 == 0 and episode_number != 0:
        #    torch.save(agent.policy.state_dict(), "PPO_VFINAL%s_%d.mdl" % (env_name, episode_number))

        # Policy update at the end of episode
        #agent.update_policy(episode_number, episode_done=done)

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
    #torch.save(agent.policy.state_dict(), "modelPPOservetousACTION0_%s_%d.mdl" % (env_name, train_run_id))
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
            action, _ = agent.get_action(observation, evaluation=False)
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
    parser.add_argument('--wandb-project-name', type=str, default="PPO_RL_Agent",
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
