############################################
# @Author: Git-123-Hub
# @Date: 2021/10/25
# @Description: implementation of PPO(proximal policy optimization)
############################################
from copy import deepcopy

import gym
import torch
import torch.nn.functional as F
from torch import optim

from utils import Agent, EpisodicReplayMemory


class PPO(Agent):

    def __init__(self, env: gym.Env, actor, critic, config: dict):
        super().__init__(env, config)
        self.state_value, self.log_prob = None, None
        # these two value should also be store in replay memory for learning

        self._actor = actor
        self.actor, self.actor_optimizer = None, None

        self._critic = critic
        self.critic, self.critic_optimizer = None, None

        self.replayMemory = EpisodicReplayMemory(self.config.get('discount_factor', 0.99))

    def run_reset(self):
        super().run_reset()
        self.actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 0.001))

        self.critic = self._critic(self.state_dim, self.config.get('critic_hidden_layer'))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get('learning_rate', 0.001))

    def select_action(self):
        # todo: with torch.no_grad()
        with torch.no_grad():
            state = torch.tensor(self.state).float().unsqueeze(0)
            action_dist = self.actor(state)
            self.action = action_dist.sample()
            self.log_prob = action_dist.log_prob(self.action)
            self.action = self.action.detach().numpy()
            self.state_value = self.critic(state)

    def save_experience(self):
        """during the procedure of training, only store (state)"""
        experience = (self.state, self.action, self.reward, self.state_value, self.log_prob)
        self.replayMemory.add(experience)

    def learn(self):
        if not self.done:  # only learn when this episode terminates
            return
        states, actions, rewards, state_values, old_log_probs, advantages, discount_rewards = self.replayMemory.fetch()
        for _ in range(self.config.get('training_epoch', 50)):
            # update actor
            dist = self.actor(states)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            clip_ratio = torch.clip(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio'])
            loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            self.logger.info(f'actor loss: {loss.item()}')
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            # update critic
            state_values = self.critic(states)
            loss = F.mse_loss(state_values, discount_rewards, reduction='mean')
            self.logger.info(f'critic loss: {loss.item()}')
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

    def save_policy(self):
        pass

    def load_policy(self, file):
        pass

    def test_action(self, state):
        pass
