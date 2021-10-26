############################################
# @Author: Git-123-Hub
# @Date: 2021/9/17
# @Description: implementation of REINFORCE(Monte-Carlo Policy gradient)
############################################
import os

import torch
from torch import optim

from utils import Agent
from utils.util import discount_sum


class REINFORCE(Agent):
    def __init__(self, env, policy, config):
        super(REINFORCE, self).__init__(env, config)
        self._policy = policy  # constructor of policy network
        self.policy, self.optimizer = None, None

        self.episode_log_prob = []  # log probability of each action in a whole episode, used to update policy
        self.episode_reward = []  # reward of each step in a whole episode, used to calculate sum of discounted reward

    def run_reset(self):
        super(REINFORCE, self).run_reset()
        self.policy = self._policy(self.state_dim, self.action_dim, self.config.get('policy_hidden_layer'))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get('learning_rate', 0.001))

    def episode_reset(self):
        super(REINFORCE, self).episode_reset()
        self.episode_log_prob, self.episode_reward = [], []

    def select_action(self):
        state = torch.tensor(self.state).float().unsqueeze(0)
        self.action, log_prob = self.policy(state)
        self.episode_log_prob.append(log_prob)

    def learn(self):
        # NOTE that REINFORCE only learn when an episode finishes
        # before that, we need to collect sequence of reward of this episode
        self.episode_reward.append(self.reward)

        if not self.done:  # only learn when an episode finishes
            return

        returns = discount_sum(self.episode_reward, self.config.get('discount_factor', 0.99), normalize=True)

        # update policy
        loss = torch.cat(self.episode_log_prob) * torch.from_numpy(-returns)
        # Note the negative sign for `returns` to change gradient ascent to gradient descent
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_policy(self):
        """save the parameter of the policy-network when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.policy.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.policy is None:
            self.policy = self._policy(self.state_dim, self.action_dim, self.config.get('policy_hidden_layer'))
        self.policy.load_state_dict(torch.load(file))
        self.policy.eval()

    def test_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad(): action, _ = self.policy(state)
        return action
