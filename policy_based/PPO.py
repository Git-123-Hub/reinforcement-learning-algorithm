############################################
# @Author: Git-123-Hub
# @Date: 2021/10/25
# @Description: implementation of PPO(proximal policy optimization)
############################################
import os

import gym
import torch
import torch.nn.functional as F
from torch import optim, nn

from utils import Agent, EpisodicReplayMemory
from utils.const import Config
from utils.replayMemoryEnhanced import ReplayMemoryEnhanced


class PPO(Agent):

    def __init__(self, env: gym.Env, actor, critic, config: Config):
        super().__init__(env, config)
        self.state_value, self.log_prob = None, None
        # these two value should also be stored in replay memory for learning

        self._actor = actor
        self.actor, self.actor_optimizer = None, None

        self._critic = critic
        self.critic, self.critic_optimizer = None, None

        self.replayMemory = ReplayMemoryEnhanced(self.config.memory_capacity, self.config.batch_size,
                                                 self.config.gamma)
        self.update_N = 0
        print(
            f'update each time: {self.config.training_epoch * self.config.memory_capacity // self.config.batch_size}')

    def run_reset(self):
        super().run_reset()
        print(f'update num: {self.update_N}')
        self.update_N = 0
        self.actor = self._actor(self.state_dim, self.action_dim, self.config.actor_hidden_layer,
                                 activation=self.config.actor_activation, max_action=self.max_action,
                                 fix_std=self.config.fix_std)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)

        self.critic = self._critic(self.state_dim, self.config.critic_hidden_layer,
                                   activation=self.config.critic_activation)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    def select_action(self):
        with torch.no_grad():
            state = torch.tensor(self.state).float().unsqueeze(0)
            action_dist = self.actor(state)
            self.action = action_dist.sample()
            self.log_prob = action_dist.log_prob(self.action).sum()  # just a tensor
            self.action = self.action.numpy()
            self.state_value = self.critic(state).squeeze()  # just a tensor

    def save_experience(self):
        """during the procedure of training, store trajectory for future learning"""
        self.replayMemory.add(self.state, self.action, self.reward, self.next_state, self.done, self.log_prob,
                              self.state_value)

    def learn(self):
        # todo: mini batch_size
        if not len(
                self.replayMemory) >= self.config.memory_capacity:  # only learn when the replay buffer if full
            return
        self.update_N += 1
        for _ in range(self.config.training_epoch * self.config.memory_capacity // self.config.batch_size):
            states, actions, rewards, next_states, dones, old_log_probs, \
            state_value, advantages, discount_rewards = self.replayMemory.sample()
            # update actor
            dist = self.actor(states)
            log_probs = dist.log_prob(actions).sum(dim=1)  # length: episode_length
            ratio = torch.exp(log_probs - old_log_probs)
            self.logger.info(
                f'ratio mean: {ratio.mean()}, std: {ratio.std()}, min: {ratio.min()}, max: {ratio.max()}')

            clip_ratio = torch.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            self.logger.info(f'actor loss: {loss.item()}')
            self.actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # update critic
            state_values = self.critic(states).squeeze()  # length: episode_length
            state_values = (state_values - state_values.mean()) / (state_values.std() + 1e-7)
            loss = F.mse_loss(state_values, discount_rewards, reduction='mean')
            self.logger.info(f'critic loss: {loss.item()}')
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        self.replayMemory.reset()

    def save_policy(self):
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.actor is None:
            self.actor = self._actor(self.state_dim, self.action_dim, self.config.actor_hidden_layer)
        self.actor.load_state_dict(torch.load(file))
        self.actor.eval()

    def test_action(self, state):
        with torch.no_grad():
            state = torch.tensor(self.state).float().unsqueeze(0)
            action_dist = self.actor(state)
            return action_dist.sample().numpy()
