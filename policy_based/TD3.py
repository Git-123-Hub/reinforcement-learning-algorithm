############################################
# @Author: Git-123-Hub
# @Date: 2021/9/28
# @Description: implementation of TD3(Twin Delayed Deep Deterministic policy gradient)
############################################
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils import Agent, replayMemory
from utils.util import soft_update


class TD3(Agent):
    def __init__(self, env, actor, critic, config):
        # todo: TD3 might use two different critic
        super(TD3, self).__init__(env, config)
        # initial actor
        self._actor = actor
        self.actor, self.target_actor, self.actor_optimizer = None, None, None

        # initial critic
        self._critic = critic
        self.critic1, self.target_critic1, self.critic1_optimizer = None, None, None
        self.critic2, self.target_critic2, self.critic2_optimizer = None, None, None

        self.replayMemory = replayMemory(self.config.memory_capacity, self.config.batch_size)

    def run_reset(self):
        super(TD3, self).run_reset()

        self.actor = self._actor(self.state_dim, self.action_dim, self.config.actor_hidden_layer,
                                 max_action=self.max_action)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)

        self.critic1 = self._critic(self.state_dim, self.action_dim, self.config.critic_hidden_layer)
        self.target_critic1 = deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.learning_rate)

        self.critic2 = self._critic(self.state_dim, self.action_dim, self.config.critic_hidden_layer)
        self.target_critic2 = deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.learning_rate)

    def select_action(self):
        self.actor.eval()
        state = torch.tensor(self.state).float().unsqueeze(0)
        self.action = self.actor(state).detach().squeeze(0).numpy()
        self.actor.train()

        # gym environment will do the clip on action
        self.action += self.get_action_noise(self.action.size)

    def get_action_noise(self, size):
        noise_mean = self.config.noise_mean
        noise_std = self.config.noise_std * self.max_action
        noise_clip = self.config.noise_clip
        noise_factor = self.config.noise_factor * self.max_action
        noise = np.random.normal(noise_mean, noise_std, size=size) * noise_factor
        return np.clip(noise, noise_clip * self.min_action, noise_clip * self.max_action)

    def learn(self):
        if not self.replayMemory.ready:  # only start to learn when there are enough experience to sample
            return
        states, actions, rewards, next_states, dones = self.replayMemory.sample()

        # get next_action of next_state using target_actor
        next_actions = self.target_actor(next_states)
        next_actions += torch.tensor(self.get_action_noise(next_actions.size()))
        # we should clip the action here, because these actions are not passed into the env
        next_actions.clip_(self.min_action, self.max_action)

        # calculate target value
        target_critic1_value = self.target_critic1(next_states, next_actions)
        target_critic2_value = self.target_critic2(next_states, next_actions)
        target_critic_value = torch.min(target_critic1_value, target_critic2_value).detach()
        target_value = rewards + self.config.gamma * target_critic_value * (1 - dones)

        # update critic1
        self.critic1.eval()
        current_value = self.critic1(states, actions)
        self.critic1.train()

        loss = F.mse_loss(current_value, target_value, reduction='mean')
        self.critic1_optimizer.zero_grad()
        loss.backward()
        self.critic1_optimizer.step()

        # update critic2
        self.critic2.eval()
        current_value = self.critic2(states, actions)
        self.critic2.train()

        loss = F.mse_loss(current_value, target_value, reduction='mean')
        self.critic2_optimizer.zero_grad()
        loss.backward()
        self.critic2_optimizer.step()

        if self.length[self._run].sum() % self.config.update_interval == 0:
            # update actor using the critic value of current state with action from actor
            loss = -self.critic1(states, self.actor(states)).mean()  # note the negative sign
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            # update target network
            soft_update(self.actor, self.target_actor, self.config.tau)
            soft_update(self.critic1, self.target_critic1, self.config.tau)
            soft_update(self.critic2, self.target_critic2, self.config.tau)

    def save_policy(self):
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.actor is None:
            self.actor = self._actor(self.state_dim, self.action_dim, self.config.actor_hidden_layer,
                                     max_action=self.max_action)
        self.actor.load_state_dict(torch.load(file))
        self.actor.eval()

    def test_action(self, state):
        # no noise added
        return self.actor(state).detach().squeeze(0).numpy()
