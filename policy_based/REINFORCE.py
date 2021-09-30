############################################
# @Author: Git-123-Hub
# @Date: 2021/9/17
# @Description: implementation of REINFORCE(Monte-Carlo Policy gradient)
############################################
import os

import gym
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt

from utils import Agent
from utils.const import Color


class REINFORCE(Agent):
    def __init__(self, env, policyNet, config):
        super(REINFORCE, self).__init__(env, config)
        self._policyNet = policyNet
        self.policy = self._policyNet(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get('learning_rate', 0.01))

    def run_reset(self):
        super(REINFORCE, self).run_reset()
        self.policy = self._policyNet(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get('learning_rate', 0.01))

    def run_an_episode(self):
        episode_log_prob, episode_reward = [], []
        while not self.done:
            self.length[self._run][self._episode] += 1

            # select action
            state = torch.tensor(self.state).float().unsqueeze(0)
            self.action, log_prob = self.policy(state)
            episode_log_prob.append(log_prob)

            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            # self.env.render()
            episode_reward.append(self.reward)
            self.rewards[self._run][self._episode] += self.reward
            self.state = self.next_state

        self.learn(episode_reward, episode_log_prob)  # learn when an episode finishes

        print(f'\r{self._episode + 1: >4}th episode: '
              f'{self.length[self._run][self._episode]: >3} steps, '
              f'rewards: {self.rewards[self._run][self._episode]: >7.2f}, '
              f'running reward: {self._running_reward: >7.2f}, '
              f'learning rate: {self._learning_rate: >7.6f}, ', end='')

    def learn(self, reward_list, log_prob_list):
        returns = np.zeros_like(reward_list)
        eps = np.finfo(np.float32).eps.item()  # tiny non-negative number
        R = 0
        for index in reversed(range(len(reward_list))):
            R = reward_list[index] + self.config.get('discount_factor', 0.99) * R
            returns[index] = R
        returns = (returns - returns.mean()) / (returns.std() + eps)
        loss_list = torch.cat(log_prob_list) * torch.from_numpy(-returns)
        # Note the negative sign for `returns` to change gradient ascent to gradient decent

        loss = loss_list.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_policy(self):
        """save the parameter of the policy network(`self.policy`) when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.policy.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        self.policy.load_state_dict(torch.load(file))
        self.policy.eval()

    def test_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad(): action, _ = self.policy(state)
        return action
