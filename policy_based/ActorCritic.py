############################################
# @Author: Git-123-Hub
# @Date: 2021/9/22
# @Description: implementation of Actor-Critic
############################################

# Note that I use parameterized value function as Critic

import os

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from utils.Agent import Agent


class ActorCritic(Agent):
    def __init__(self, env, actor, critic, config):
        super(ActorCritic, self).__init__(env, config)
        self._actor = actor
        self.actor = self._actor(self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 0.001))

        self._critic = critic
        self.critic = self._critic(self.state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get('learning_rate', 0.001))

    def run_reset(self):
        super(ActorCritic, self).run_reset()
        self.actor = self._actor(self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 0.001))

        self.critic = self._critic(self.state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get('learning_rate', 0.001))

    def run_an_episode(self):
        rewards, log_probs, state_values = [], [], []
        while not self.done:
            # select action
            self.action, log_prob = self.actor(self.state)

            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            # self.env.render()

            self.length[self._run][self._episode] += 1
            self.rewards[self._run][self._episode] += self.reward

            rewards.append(self.reward)
            log_probs.append(log_prob)
            state_values.append(self.critic(self.state))

            self.learn(rewards, log_probs, state_values)

            self.state = self.next_state

        print(f'\r{format(self._episode + 1, ">3")}th episode: '
              f'{format(self.length[self._run][self._episode], ">3")} steps, '
              f'rewards: {format(self.rewards[self._run][self._episode], ">5.1f")}, '
              f'running reward: {format(self._running_reward, ">7.3f")}, ', end='')

    def learn(self, reward_list, log_prob_list, state_value_list):
        if self.done:  # learn after this episode finishes
            # calculate the true value using rewards returned from the environment
            returns = np.zeros_like(reward_list)
            eps = np.finfo(np.float32).eps.item()  # tiny non-negative number
            R = 0
            for index in reversed(range(len(reward_list))):
                R = reward_list[index] + self.config.get('discount_factor', 0.99) * R
                returns[index] = R
            returns = (returns - returns.mean()) / (returns.std() + eps)
            returns = torch.tensor(returns, dtype=torch.float)

            advantage_list = returns - torch.cat(state_value_list).squeeze()
            policy_loss_list = torch.cat(log_prob_list) * -advantage_list.detach()
            value_loss_list = F.mse_loss(torch.cat(state_value_list).squeeze(), returns, reduction='none')

            # update actor
            self.actor_optimizer.zero_grad()
            loss = policy_loss_list.sum()
            loss.backward()
            self.actor_optimizer.step()

            # update critic
            self.critic_optimizer.zero_grad()
            loss = value_loss_list.sum().float()
            loss.backward()
            self.critic_optimizer.step()

    def save_policy(self):
        """save the parameter of the policy network(`self.Q) when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        self.actor.load_state_dict(torch.load(file))
        self.actor.eval()

    def test_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad(): action, _ = self.actor(state)
        return action
