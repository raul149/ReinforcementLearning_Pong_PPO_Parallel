import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent_ppo_parallel import Agent
from wimblepong import Wimblepong
from parallel_env import ParallelEnvs
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_timesteps=1000000, update_steps=400):
    # Create a Gym environment
    # This creates 30 parallel envs running in 6 processes (5 envs each)
    env = ParallelEnvs(env_name, processes=6, envs_per_process=5)

    #Wandb modules, to visualize
    exp_name = 'PPO_Model_v2'
    experiment_name = exp_name
    data_path = os.path.join('data', experiment_name)
    models_path = f"{data_path}/models"
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=exp_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{exp_name}")

    #Create agent and load model
    agent = Agent()
    agent.load_model()

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    # Reset the environment and observe the initial state
    observation = env.reset()
    #Counter to see WR during trainign as well
    wonp=0
    lost=0

    # Loop forever
    for timestep in range(train_timesteps):
        # Get action from the agent
        #print(observation.shape)
        action, action_probabilities = agent.get_action(observation)
        previous_observation = observation

        # Perform the action on the environment, get new state and reward
        observation, reward, done, info = env.step(action)

        for i in range(len(info["infos"])):
            env_done = False
            # Check if the environment is finished; if so, store cumulative reward
            for envid, envreward in info["finished"]:
                if envid == i:
                    reward_history.append(envreward)
                    average_reward_history.append(np.mean(reward_history[-500:]))
                    env_done = True
                    #Scalar for WANDB
                    writer.add_scalar('Training Reward' + env_name, reward[i], timestep)
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
            agent.store_outcome(previous_observation[i], observation[i], action[i],
                                action_probabilities[i], reward[i], env_done)



        if timestep % update_steps == update_steps-1:
            print(f"Update @ step {timestep}")
            agent.update_policy(0)


        plot_freq = 1500
        if timestep % plot_freq == plot_freq-1:
            # Training is finished - plot rewards
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "500-episode average"])
            plt.title("AC reward history (non-episodic, parallel)")
            plt.savefig("Rewards_Model_Version2_PPOLArger_%s.png" % env_name)
            plt.clf()
            update_steps=update_steps+1
            if update_steps>520:
                update_steps=520
        model_freq=10500
        if timestep % model_freq == 0 :
            torch.save(agent.policy.state_dict(), "Model_Version2_PPOLArgerModel_Scratchdumbestai_%s_%d.mdl" % (env_name, timestep))
            print("%d: Plot and model saved." % timestep)
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Nonepisodic parallel AC"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "Model_Version2_PPOLArger_%s_%d.mdl" % (env_name, train_run_id))
    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0", help="Environment to use")
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

