############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: implementation of DDPG(Deep Deterministic Policy Gradient)
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


class DDPG(Agent):
    def __init__(self, env, actor, critic, config):
        super(DDPG, self).__init__(env, config)
        self._actor = actor
        self._critic = critic

        # these are initialized in `run_reset()`
        self.actor, self.target_actor, self.actor_optimizer = None, None, None
        self.critic, self.target_critic, self.critic_optimizer = None, None, None

        self.replayMemory = replayMemory(self.config.get('memory_capacity', 20000), self.config.get('batch_size', 256))

        # todo: maybe need a random process

    def run_reset(self):
        super(DDPG, self).run_reset()
        self.replayMemory.reset()

        self.actor = self._actor(self.state_dim)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 0.001))

        self.critic = self._critic(self.state_dim, self.action_dim)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get('learning_rate', 0.001))

    def run_an_episode(self):
        while not self.done:
            self.length[self._run][self._episode] += 1
            self.select_action()

            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            self.env.render()

            self.rewards[self._run][self._episode] += self.reward

            # save experience
            experience = (self.state, self.action, self.reward, self.next_state, self.done)
            self.replayMemory.add(experience)

            # only start to learn when there are enough experiences to sample from
            self.learn()

            self.state = self.next_state

        print(f'\r{format(self._episode + 1, ">3")}th episode: '
              f'{format(self.length[self._run][self._episode], ">3")} steps, '
              f'rewards: {format(self.rewards[self._run][self._episode], ">5.1f")}, '
              f'running reward: {format(self._running_reward, ">7.3f")}', end='')

    def select_action(self):
        self.actor.eval()
        with torch.no_grad():
            # todo: check data dimension
            self.action = self.actor(self.state).squeeze(0).numpy()
        self.actor.train()

        noise = np.random.normal(0, 1)

        self.action += noise  # gym environment will do the clip on action

    def learn(self):
        if self.replayMemory.ready:
            self._states, self._actions, self._rewards, self._next_states, self._dones = self.replayMemory.sample()
            loss = F.mse_loss(self.current_state_action_value, self.target_value)
            self.logger.info(f'loss: {loss.item()}')

            # update critic
            self.critic_optimizer.zero_grad()
            loss = F.mse_loss(self.current_state_action_value, self.target_value)
            loss.backward()
            self.critic_optimizer.step()

            # update actor using the critic value of sampled states
            self.actor_optimizer.zero_grad()
            # note these actions are not sampled, but get from actor
            actions = self.actor(self._states)
            loss = self.critic(self._states, actions)
            loss.mean().backward()
            self.actor_optimizer.step()

            self.update_target_network()

    @property
    def current_state_action_value(self):
        return self.critic(self._states, self._actions)

    @property
    def next_state_action_value(self):
        # get next action using target_actor
        next_action = self.target_actor(self._next_states).squeeze(0)
        return self.target_critic(self._next_states, next_action).detach()

    @property
    def target_value(self):
        return self._rewards + self.config.get('discount_factor', 0.99) * self.next_state_action_value * (
                    1 - self._dones)

    def update_target_network(self):
        if self.length[self._run].sum() % self.config.get('Q_update_interval', 0) == 0:  # time to update
            if 'tau' in self.config:  # soft update
                soft_update(self.actor, self.target_actor, self.config.get('tau', 0.01))
                soft_update(self.critic, self.target_critic, self.config.get('tau', 0.01))
            else:
                self.target_actor = deepcopy(self.actor)
                self.target_critic = deepcopy(self.critic)

    def save_policy(self):
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))