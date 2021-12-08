############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: implementation of Deep-Q-Learning(NPIS 2013)
############################################
import copy
import os
import random

import torch
import torch.nn.functional as F
from torch import optim

from utils import replayMemory, Agent
from utils.const import Color
from utils.util import soft_update


class DQN(Agent):
    def __init__(self, env, Q_net, config):
        super(DQN, self).__init__(env, config)

        self._Q = Q_net  # constructor for Q network
        if hasattr(Q_net, 'dueling'):
            self.dueling = True
            print(f'{Color.INFO}using the dueling network{Color.END}')

        self.Q, self.target_Q, self.optimizer = None, None, None
        self.replay_buffer = replayMemory(self.config.memory_capacity, self.config.batch_size)

        self._epsilon = None  # decay according to episode number during training

    def run_reset(self):
        super(DQN, self).run_reset()
        self.Q = self._Q(self.state_dim, self.action_dim, self.config.q_hidden_layer)
        self.target_Q = copy.deepcopy(self.Q)
        self.optimizer = optim.Adam(self.Q.parameters(),
                                    lr=self.config.learning_rate,
                                    eps=1e-4)

    def episode_reset(self):
        super(DQN, self).episode_reset()

        # update epsilon decay
        epsilon0 = self.config.epsilon
        min_epsilon = self.config.min_epsilon
        if self._episode == 0:  # start with initial value
            self._epsilon = epsilon0
        elif self.running_rewards[self._run][self._episode - 1] >= self.goal:  # reaches goal
            self._epsilon = min_epsilon
        else:
            self._epsilon *= self.config.epsilon_decay_rate  # exponential decay
            self._epsilon = max(self._epsilon, min_epsilon)

    def select_action(self):
        """select action according to `self.Q` given current state"""
        state = torch.tensor(self.state).float().unsqueeze(0)
        if random.random() > self._epsilon:
            # some modules behave differently in training/evaluation mode, e.g. Dropout, BatchNorm
            self.Q.eval()
            with torch.no_grad():
                actions_value = self.Q(state)
            self.Q.train()

            self.action = actions_value.argmax().item()
            value_to_show = [float(format(v, '.10f')) for v in actions_value.tolist()[0]]
            self.logger.info(f'actions value: {value_to_show}, choose action: {self.action}')
        else:
            self.action = self.env.action_space.sample()
            self.logger.info(f'choose randomly, action: {self.action}')

    def learn(self):
        if not self.replay_buffer.ready:  # only start to learn when there are enough experiences to sample
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        current_state_value = self.Q(states).gather(1, actions.long())
        next_state_value = self.get_next_state_value(next_states)
        target_value = rewards + self.config.gamma * next_state_value * (1 - dones)

        loss = F.mse_loss(current_state_value, target_value)
        self.logger.info(f'loss: {loss.item()}')

        self.gradient_descent(loss)
        self.update_target_network()

    def get_next_state_value(self, next_states):
        return self.target_Q(next_states).detach().max(1)[0].unsqueeze(1)

    def gradient_descent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad = self.config.clip_grad
        if clip_grad: torch.nn.utils.clip_grad_norm_(self.Q.parameters(), clip_grad)
        self.optimizer.step()

    def update_target_network(self):
        if self.length[self._run].sum() % self.config.update_interval == 0:
            tau = self.config.tau
            if tau is not None:
                soft_update(self.Q, self.target_Q, tau)
            else:
                self.target_Q = copy.deepcopy(self.Q)

    def save_policy(self):
        """save the parameter of the Q network(`self.Q) when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.Q.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.Q is None: self.Q = self._Q(self.state_dim, self.action_dim, self.config.q_hidden_layer)
        self.Q.load_state_dict(torch.load(file))
        self.Q.eval()

    def test_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad(): actions_value = self.Q(state)
        action = actions_value.argmax().item()
        return action
