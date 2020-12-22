import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 9 * 9, 512)

        self.fc1_actor = torch.nn.Linear(512, 256)
        self.fc1_critic = torch.nn.Linear(512, 256)

        self.fc2_probs = torch.nn.Linear(256, 3)
        self.fc2_value = torch.nn.Linear(256, 1)

        self.reset_parameters()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.fc1_actor.weight.data.mul_(relu_gain)
        self.fc1_critic.weight.data.mul_(relu_gain)



    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 9 * 9)
        x = self.linear1(x)
        x = F.relu(x)


        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        x_mean = self.fc2_probs(x_ac)

        x_probs = F.softmax(x_mean, dim=-1)
        x_log_probs = F.log_softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)

        entropy = dist.entropy()

        return dist, entropy, value



class Agent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.device)
        self.previous_frame = None
        self.gamma = 0.9
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=7e-4, eps=1e-5, alpha=0.99)
        self.state_values = []
        self.rewards = []
        self.action_probs = []
        self.entropies = []
        self.timesteps = []
        self.value_loss_coef=.5
        self.entropy_coef=1e-3

    def get_action_train(self, observation):
        x = self.preprocess(observation).to(self.device)
        dist, entropy, state_value = self.policy(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, entropy, state_value

    def get_action(self, observation):
        x = self.preprocess(observation).to(self.device)
        dist, _, _ = self.policy(x)
        return torch.argmax(dist.probs)

    def reset(self):
        self.previous_frame = None

    def get_name(self):
        return "Big Fish"

    def load_model(self):
        filename = 'modelfish.mdl'
        weights = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(weights, strict=False)
        self.policy.eval()


    #FROM TA provided SomeAgent/SomeOtherAgent
    def preprocess(self, observation):
        observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.previous_frame is None:
            self.previous_frame = observation
        stack_ob = np.concatenate((self.previous_frame, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        stack_ob = stack_ob.transpose(1, 3)
        self.previous_frame = observation
        return stack_ob


    def store_outcome(self, state_value, reward, action_prob, entropy):
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.action_probs.append(action_prob)
        self.entropies.append(entropy)

    def episode_finished(self):
        state_values = torch.stack(self.state_values, dim=0).squeeze().to(self.device)
        returns = discount_rewards(torch.tensor(self.rewards, device=self.device, dtype=torch.float), self.gamma)
        action_probs = torch.stack(self.action_probs, dim=0).squeeze().to(self.device)
        entropies = torch.stack(self.entropies, dim=0).squeeze().to(self.device)
        self.state_values, self.rewards, self.action_probs, self.entropies = [], [], [], []


        advantages = returns.detach() - state_values

        policy_loss = -(advantages.detach() * action_probs).mean()
        entropy_loss = -(self.entropy_coef * entropies).mean()
        value_loss = (self.value_loss_coef*advantages.pow(2)).mean()

        loss = policy_loss + entropy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), .5)
        self.optimizer.step()
