Here you have the model file: model.mdl
The agent file used for testing: agent.py 
The agent has been trained using PPO parallel, with different AI behaviours, such as not following the ball:
Thus:
the training file: train.py, runs using 30 parallel environments(And won't run with the agent.py, since it doesn't make a list of lists)
The additional file: agent_ppo_parallel.py. Has been used to train the model for parallel environemnts as it deals with the different stack of pictures it arrives. Although we could make it work in one, it is a bit messy for us, and unclear. If that would be a problem, please let us know.
Since the model is valid for both and will work for both.
we add the simple_ai.py file with the modifications we did, so that this can be seen against which agents it was trained to uderstand the end-behaviour.
Parallel-env: Is the same file from the Assignment 6, remains untouched.
