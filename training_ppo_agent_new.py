import torch
import gym
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from agent_new import Agent
from agent import Agent as Agent2
from agent_opponent import Agent as Opponent
import wimblepong # import wimblepong-environment
import pandas as pd
from PIL import Image
from collections import deque
import os
import random
from torch.utils.tensorboard import SummaryWriter


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environmentS
    env = gym.make("WimblepongVisualMultiplayer-v0")

    exp_name = 'PPO_COMPLEXNN'
    experiment_name = exp_name
    data_path = os.path.join('data', experiment_name)
    models_path = f"{data_path}/models"
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=exp_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{exp_name}")

    agent = Agent()
    agent.load_model() # TODO: uncomment if new model should be created

    opponent = Agent2()
    opponent.load_model()


    player_id = 1
    opponent_id = 3 - player_id
    opponent = wimblepong.SimpleAi(env, opponent_id)

    env.set_names(agent.get_name(), opponent.get_name())

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    counter = 0
    best=0
    identity=0
    wrai=0
    winai=0
    lossai=0
    wrag2=0
    winag2=0
    lossag2=0
    wragf=0
    winagf=0
    lossagf=0
    wrag3=0
    winag3=0
    lossag3=0
    identity=0


    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        ob1, ob2 = env.reset()
        agent.reset()  # reset all remaining state transition buffers
        opponent.reset()
        # Loop until the episode is over
        while not done:
            # Get action from the agent, an action gets chosen based on the img_stacked processed.
            if identity!=0:

                action1, action_probabilities1 = agent.get_action(ob1,evaluation=False)
                action2 = opponent.get_action(ob2)
                previous_observation = ob1

            # Perform the action on the environment, get new state and reward
                (ob1, ob2), (rew1, rew2), done, info = env.step((action1,action2))
            else:
                action1, action_probabilities1 = agent.get_action(ob1,evaluation=False)
                action2= opponent.get_action()
                previous_observation = ob1

                # Perform the action on the environment, get new state and reward
                (ob1, ob2), (rew1, rew2), done, info = env.step((action1,action2))




            #env.render() # TODO: uncomment to test and see how it plays pong
            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, ob1, action_probabilities1, rew1, done, action1)

            # Store total episode reward
            reward_sum += rew1
            timesteps += 1
            counter += 1
        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))
            print("Opponent:", opponent.get_name())
        # Update the actor-critic code to perform TD(0) updates every 50 timesteps
        if counter > 11000:
            counter = 0
            agent.update_policy(episode_number, episode_done=done)
        writer.add_scalar('Training Reward' + env_name, reward_sum, episode_number)
        writer.add_scalar('Training Timesteps ' + env_name, timesteps, episode_number)
        if identity==0:
            if reward_sum==10:
                winai += 1
            else:
                lossai += 1
            wrai=winai/(winai+lossai)
            print('WinRate vs AI:', wrai)
        else:
            if identity==1:
                if reward_sum==10:
                    winag2 += 1
                else:
                    lossag2 += 1
                wrag2=winag2/(winag2+lossag2)
                print('WinRate vs Agentus:', wrag2)
            else:
                if identity==2:
                    if reward_sum==10:
                        winagf += 1
                    else:
                        lossagf += 1
                    wragf=winagf/(winagf+lossagf)
                    print('WinRate vs AgentBigFish:', wragf)
                else:
                    if reward_sum==10:
                        winag3 += 1
                    else:
                        lossag3 += 1
                    wrag3=winag3/(winag3+lossag3)
                    print('WinRate vs AgentDeliveredVersion:', wrag3)

        identity = random.choice([0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3])
        if identity==0:
            opponent = wimblepong.SimpleAi(env, opponent_id)
        if identity==1:
            opponent = Agent()
            opponent.load_model()
        if identity==2:
            opponent = Opponent()
            opponent.load_model()
        if identity==3:
            opponent = Agent2()
            opponent.load_model2()





        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        if episode_number % 397 == 0 and episode_number != 0:
            torch.save(agent.policy.state_dict(), "PPOLARGENETWORKV3%s_%d.mdl" % (env_name, episode_number))

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
