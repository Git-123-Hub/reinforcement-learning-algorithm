############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: implementation of DDPG(Deep Deterministic Policy Gradient)
############################################
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils import Agent, replayMemory
from utils.util import soft_update


class DDPG(Agent):
    def __init__(self, env, actor, critic, config):
        super(DDPG, self).__init__(env, config)
        self._actor = actor
        self._critic = critic

        # these are initialized in `run_reset()`
        self.actor, self.target_actor, self.actor_optimizer = None, None, None
        self.critic, self.target_critic, self.critic_optimizer = None, None, None

        self.replayMemory = replayMemory(self.config.memory_capacity, self.config.batch_size)

    def run_reset(self):
        super(DDPG, self).run_reset()

        self.actor = self._actor(self.state_dim, self.action_dim, self.config.actor_hidden_layer,
                                 max_action=self.max_action)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)

        self.critic = self._critic(self.state_dim, self.action_dim, self.config.critic_hidden_layer)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    def select_action(self):
        self.actor.eval()
        state = torch.tensor(self.state).float().unsqueeze(0)
        self.action = self.actor(state).detach().squeeze(0).numpy()
        self.actor.train()

        noise = np.random.normal(0, 1)

        self.action += noise  # gym environment will do the clip on action

    def learn(self):
        if not self.replayMemory.ready:  # only start to learn when there are enough experience to learn
            return
        states, actions, rewards, next_states, dones = self.replayMemory.sample()

        # update critic using the loss between current_state_value and bootstrap target_value
        self.critic.eval()
        current_state_action_value = self.critic(states, actions)
        self.critic.train()

        # calculate next_state_action_value using the next_action get from target_actor
        next_action = self.target_actor(next_states).detach()
        next_state_action_value = self.target_critic(next_states, next_action).detach()
        target_value = rewards + self.config.gamma * next_state_action_value * (1 - dones)

        loss = F.mse_loss(current_state_action_value, target_value, reduction='mean')
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # update actor using the critic value of sampled states
        # note these actions are not sampled, but get from actor
        actions = self.actor(states)
        loss = -self.critic(states, actions).mean()  # note the negative sign
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # update target network
        if self.length[self._run].sum() % self.config.update_interval == 0:  # time to update
            soft_update(self.actor, self.target_actor, self.config.tau)
            soft_update(self.critic, self.target_critic, self.config.tau)

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
        return self.actor(state).detach().squeeze(0).numpy()
