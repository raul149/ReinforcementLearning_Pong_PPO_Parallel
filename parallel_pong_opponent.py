import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent_ppo_parallel import Agent, Policy
from agent_ppo import Agent as AgentOp
from agent_ppo import Policy as PolicyOp
from wimblepong import Wimblepong
from parallel_env import ParallelEnvs
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_timesteps=1000000, update_steps=470):
    # Create a Gym environment
    # This creates 64 parallel envs running in 8 processes (8 envs each)
    env = ParallelEnvs(env_name, processes=5, envs_per_process=4)
    exp_name = 'PPO_Model_Parallel_Final_AgainstAgent'
    experiment_name = exp_name
    data_path = os.path.join('data', experiment_name)
    models_path = f"{data_path}/models"
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=exp_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{exp_name}")

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape
    observation_space_dim = env.observation_space.shape



    # Set the names for both SimpleAIs


    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)
    agent.load_model()

    policy2 = Policy(observation_space_dim, action_space_dim)
    agent2 = Agent(policy2)
    agent2.load_model()

    player_id = 1
    opponent_id = 3 - player_id
    print(env_name)


    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    # Reset the environment and observe the initial state
    ob2 = env.reset()
    ob1 = ob2[:,1,:,:,:]
    ob2 = ob1
    print(ob1.shape)
    print(ob2.shape)
    wonp=0
    lost=0

    # Loop forever
    for timestep in range(train_timesteps):
        # Get action from the agent
        #print(observation.shape)
        action1, action_probabilities = agent.get_action(ob1)
        action2, action_probabilities = agent2.get_action(ob2)
        previous_observation = ob1

        # Perform the action on the environment, get new state and reward
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1.detach().cpu().numpy(),action2.detach().cpu().numpy()))

        for i in range(len(info["infos"])):
            env_done = False
            # Check if the environment is finished; if so, store cumulative reward
            for envid, envreward in info["finished"]:
                if envid == i:
                    reward_history.append(envreward)
                    average_reward_history.append(np.mean(reward_history[-500:]))
                    env_done = True
                    writer.add_scalar('Training Reward' + env_name, rew1[i], timestep)
                    agent.reset(i)
                    print('Episode finished',i,'timestep', timestep)
                    if envreward==10:
                        wonp=wonp+1
                    if envreward==-10:
                        lost=lost+1
                        print("Winning Rate:", wonp/(wonp+lost))
                        print("Won:" ,wonp)
                        print("Lost:" ,lost)
                    break
            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation[i], ob1[i], action1[i],
                                action_probabilities[i], rew1[i], env_done)



        if timestep % update_steps == update_steps-1:
            print(f"Update @ step {timestep}")
            agent.update_policy(0)


        plot_freq = 1000
        if timestep % plot_freq == plot_freq-1:
            # Training is finished - plot rewards
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "500-episode average"])
            plt.title("AC reward history (non-episodic, parallel)")
            plt.savefig("Rewards_Model_Parallel_Final_AgainstAgent_%s.png" % env_name)
            plt.clf()
        model_freq=1500
        if timestep % model_freq == 0 :
            torch.save(agent.policy.state_dict(), "Model_Parallel_Final_AgainstAgent_%s_%d.mdl" % (env_name, timestep))
            print("%d: Plot and model saved." % timestep)
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Nonepisodic parallel AC"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "Model_Parallel_Final_AgainstAgent_%s_%d.mdl" % (env_name, train_run_id))
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
    parser.add_argument("--env", type=str, default="WimblepongVisualMultiplayer-v0", help="Environment to use")
    parser.add_argument("--train_timesteps", type=int, default=2000000, help="Number of timesteps to train for")
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
            train(args.env, train_timesteps=args.train_timesteps)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

