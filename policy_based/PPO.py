############################################
# @Author: Git-123-Hub
# @Date: 2021/10/25
# @Description: implementation of PPO(proximal policy optimization)
############################################
import os

import gym
import torch
import torch.nn.functional as F
from torch import optim

from utils import Agent, EpisodicReplayMemory


class PPO(Agent):

    def __init__(self, env: gym.Env, actor, critic, config: dict):
        super().__init__(env, config)
        self.state_value, self.log_prob = None, None
        # these two value should also be stored in replay memory for learning

        self._actor = actor
        self.actor, self.actor_optimizer = None, None

        self._critic = critic
        self.critic, self.critic_optimizer = None, None

        self.replayMemory = EpisodicReplayMemory(self.config.get('discount_factor', 0.99))

    def run_reset(self):
        super().run_reset()
        self.actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 1e-3))

        self.critic = self._critic(self.state_dim, self.config.get('critic_hidden_layer'),
                                   activation=self.config.get('critic_activation'))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get('learning_rate', 1e-3))

    def select_action(self):
        with torch.no_grad():
            state = torch.tensor(self.state).float().unsqueeze(0)
            action_dist = self.actor(state)
            self.action = action_dist.sample()
            self.log_prob = action_dist.log_prob(self.action).sum()
            self.action = self.action.numpy()
            self.state_value = self.critic(state).squeeze(dim=0)  # shape: torch.Size([1])

    def save_experience(self):
        """during the procedure of training, store trajectory for future learning"""
        experience = (self.state, self.action, self.reward, self.state_value, self.log_prob)
        self.replayMemory.add(experience)

    def learn(self):
        if not self.done:  # only learn when this episode terminates
            return
        states, actions, rewards, _, old_log_probs, advantages, discount_rewards = self.replayMemory.fetch()
        for _ in range(self.config.get('training_epoch', 50)):
            # update actor
            dist = self.actor(states)
            log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)  # torch.size([batch_size, 1])
            ratio = torch.exp(log_probs - old_log_probs)
            self.logger.info(f'ratio mean: {ratio.mean()}, std: {ratio.std()}, min: {ratio.min()}, max: {ratio.max()}')

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
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.actor is None:
            self.actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'))
        self.actor.load_state_dict(torch.load(file))
        self.actor.eval()

    def test_action(self, state):
        with torch.no_grad():
            state = torch.tensor(self.state).float().unsqueeze(0)
            action_dist = self.actor(state)
            return action_dist.sample().numpy()
