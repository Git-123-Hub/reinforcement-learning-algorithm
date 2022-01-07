############################################
# @Author: Git-123-Hub
# @Date: 2021/10/25
# @Description: replay memory that handle trajectory of one episode each time
############################################

import numpy as np
import torch

from utils.util import discount_sum


class EpisodicReplayMemory:
    """ data structure of replay memory that only stores the trajectory of current episode """

    def __init__(self, gamma):
        self.gamma = gamma
        # following variables are added during training
        self.states, self.actions, self.rewards, self.state_values, self.log_probs = [], [], [], [], []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """clear all the experience that has been stored"""
        self.states, self.actions, self.rewards, self.state_values, self.log_probs = [], [], [], [], []

    def add(self, state, action, reward, state_value, log_prob):
        """
        store the experience at `episode_trajectory`
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.log_probs.append(log_prob)

    def fetch(self):
        """fetch all the data in the `memory` and transfer them to tensor for learning"""

        # NOTE that `states` and `actions` will be passed to network
        # so the first dim should be batch_size, i.e. episode_length
        states = torch.from_numpy(np.vstack(self.states)).float().to(self.device)  # torch.Size([length, state_dim])
        actions = torch.from_numpy(np.vstack(self.actions)).float().to(self.device)  # torch.Size([length, action_dim])
        rewards = torch.tensor(self.rewards).float()  # tensor length: episode_length
        state_values = torch.stack(self.state_values).float().to(self.device)  # tensor length: episode_length
        log_probs = torch.stack(self.log_probs).float().to(self.device)  # tensor length: episode_length

        discount_rewards = discount_sum(self.rewards, self.gamma)
        # discount_rewards = torch.from_numpy(np.vstack(self.discount_rewards)).float()
        discount_rewards = torch.tensor(discount_rewards).float()  # tensor length: episode_length
        # ?? a) calculate advantage
        # advantages = discount_rewards - state_values
        # ?? b) GAE
        gamma, lam = 0.99, 0.95
        self.rewards.append(0)
        self.state_values.append(0)

        deltas = np.array(self.rewards)[:-1] + \
                 gamma * np.array(self.state_values)[1:] - np.array(self.state_values)[:-1]
        advantages = discount_sum(deltas, gamma * lam, normalize=True)
        advantages = torch.from_numpy(advantages).float()  # tensor length: episode_length

        self.reset()
        return states, actions, rewards, state_values, log_probs, advantages, discount_rewards

    def __len__(self):
        return len(self.states)
