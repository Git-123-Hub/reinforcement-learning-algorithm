############################################
# @Author: Git-123-Hub
# @Date: 2021/9/28
# @Description: implementation of TD3(Twin Delayed Deep Deterministic policy gradient)
############################################
import os
from copy import deepcopy

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from utils.Agent import Agent
from utils.replayMemory import replayMemory
from utils.util import soft_update


class TD3(Agent):
    def __init__(self, env, actor, critic, config):
        super(TD3, self).__init__(env, config)
        # initial actor
        self._actor = actor
        self.actor, self.target_actor, self.actor_optimizer = None, None, None

        # initial critic
        self._critic = critic
        self.critic1, self.target_critic1, self.critic1_optimizer = None, None, None
        self.critic2, self.target_critic2, self.critic2_optimizer = None, None, None

        self.replayMemory = replayMemory(self.config.get('memory_capacity', 20000), self.config.get('batch_size', 256))

        self.max_action = self.env.action_space.high[0]
        self.min_action = self.env.action_space.low[0]

    def run_reset(self):
        super(TD3, self).run_reset()
        self.replayMemory.reset()

        self.actor = self._actor(self.state_dim, self.action_dim, max_action=self.max_action)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 0.001))

        self.critic1 = self._critic(self.state_dim, self.action_dim)
        self.target_critic1 = deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.get('learning_rate', 0.001))

        self.critic2 = self._critic(self.state_dim, self.action_dim)
        self.target_critic2 = deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.get('learning_rate', 0.001))

    def run_an_episode(self):
        while not self.done:
            self.select_action()

            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            # self.env.render()

            self.length[self._run][self._episode] += 1
            self.rewards[self._run][self._episode] += self.reward

            # save experience
            experience = (self.state, self.action, self.reward, self.next_state, self.done)
            self.replayMemory.add(experience)

            self.learn()

            self.state = self.next_state

        print(f'\repisode: {format(self._episode + 1, ">3")}, '
              f'steps: {format(self.length[self._run][self._episode], ">3")}, '
              f'rewards: {format(self.rewards[self._run][self._episode], ">5.1f")}, '
              f'running reward: {format(self._running_reward, ">7.3f")}', end='')

    def select_action(self):
        self.actor.eval()
        self.action = self.actor(self.state).detach().squeeze(0).numpy()
        self.actor.train()

        # gym environment will do the clip on action
        self.action += self.get_action_noise(self.action.size)

    def get_action_noise(self, size):
        noise_mean = self.config.get('noise_mean', 0)
        noise_std = self.config.get('noise_std', 1) * self.max_action
        noise_clip = self.config.get('noise_clip', 1)
        noise = np.random.normal(noise_mean, noise_std, size=size) * 0.2
        return np.clip(noise, noise_clip * self.min_action, noise_clip * self.max_action)

    def learn(self):
        if not self.replayMemory.ready:
            return
        self._states, self._actions, self._rewards, self._next_states, self._dones = self.replayMemory.sample()

        # get next action using target actor and next state
        next_actions = self.target_actor(self._next_states)
        next_actions += torch.tensor(self.get_action_noise(next_actions.size()))
        # we should clip the action here, because these actions are not passed into the env
        next_actions.clip(self.min_action, self.max_action)

        # calculate target value
        target_critic1_value = self.target_critic1(self._next_states, next_actions)
        target_critic2_value = self.target_critic2(self._next_states, next_actions)
        target_critic_value = torch.min(target_critic1_value, target_critic2_value).detach()
        target_value = self._rewards + self.config.get(
            'discount_factor', 0.99) * target_critic_value * (1 - self._dones)

        # update critic1
        self.critic1_optimizer.zero_grad()

        self.critic1.eval()
        current_value = self.critic1(self._states, self._actions)
        self.critic1.train()

        loss = F.mse_loss(current_value, target_value, reduction='mean')
        loss.backward()
        self.critic1_optimizer.step()

        # update critic2
        self.critic2_optimizer.zero_grad()

        self.critic2.eval()
        current_value = self.critic2(self._states, self._actions)
        self.critic2.train()

        loss = F.mse_loss(current_value, target_value, reduction='mean')
        loss.backward()
        self.critic2_optimizer.step()

        # time to update actor and target network
        if self.length[self._run].sum() % self.config.get('update_interval', 1) == 0:
            loss = -self.critic1(self._states, self.actor(self._states)).mean()  # note the negative sign
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor, self.target_actor, self.config.get('tau', 0.01))
            soft_update(self.critic1, self.target_critic1, self.config.get('tau', 0.01))
            soft_update(self.critic2, self.target_critic2, self.config.get('tau', 0.01))

    def save_policy(self):
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))
