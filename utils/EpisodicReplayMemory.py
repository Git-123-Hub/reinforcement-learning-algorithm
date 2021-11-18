############################################
# @Author: Git-123-Hub
# @Date: 2021/10/25
# @Description: replay memory that handle trajectory of one episode each time
############################################
from copy import deepcopy

import numpy as np
import torch

from utils.util import discount_sum


class EpisodicReplayMemory:
    """ data structure of replay memory that only stores the trajectory of current episode """

    def __init__(self, gamma):
        self.gamma = gamma
        # following variables should be specified when adding
        self.states, self.actions, self.rewards, self.state_values, self.log_probs = [], [], [], [], []
        self.advantages, self.discount_rewards = [], []  # these variables are calculated after episode terminates

    def reset(self):
        """clear all the experience that has been stored"""
        self.states, self.actions, self.rewards, self.state_values, self.log_probs = [], [], [], [], []
        self.advantages, self.discount_rewards = [], []

    def add(self, experience):
        """
        store the experience at `episode_trajectory`
        :param experience: each experience is a tuple (state, action, reward, state_value, log_prob)
        """
        state, action, reward, state_value, log_prob = experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.log_probs.append(log_prob)

    def fetch(self):
        """fetch all the data in the `memory` and transfer them to tensor for learning"""
        states = torch.from_numpy(np.vstack(self.states)).float()
        actions = torch.from_numpy(np.vstack(self.actions)).float()
        rewards = torch.from_numpy(np.vstack(self.rewards)).float()
        # state_values = torch.from_numpy(np.vstack(self.state_values)).float()
        # log_probs = torch.from_numpy(np.vstack(self.log_probs)).float()
        state_values = torch.stack(self.state_values).float()
        log_probs = torch.stack(self.log_probs).float()

        self.discount_rewards = discount_sum(self.rewards, self.gamma)
        # discount_rewards = torch.from_numpy(np.vstack(self.discount_rewards)).float()
        discount_rewards = torch.tensor(self.discount_rewards).float()
        # ?? a) calculate advantage
        # advantages = discount_rewards - state_values
        # ?? b) GAE
        gamma, lam = 0.99, 0.95
        # r = deepcopy(self.rewards)
        # r.append(0)
        # v = deepcopy(self.state_values)
        # v.append(0)

        self.rewards.append(0)
        self.state_values.append(0)

        deltas = np.array(self.rewards)[:-1] + gamma * np.array(self.state_values)[1:] - np.array(self.state_values)[
                                                                                         :-1]
        advantages = discount_sum(deltas, gamma * lam)
        advantages = torch.from_numpy(np.vstack(advantages)).float()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        self.reset()
        return states, actions, rewards, state_values, log_probs, advantages, discount_rewards

    def __len__(self):
        return len(self.states)
