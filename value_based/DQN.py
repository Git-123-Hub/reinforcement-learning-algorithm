############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: implementation of Deep-Q-Learning(NPIS 2013)
############################################
import copy
import os
import random
import time
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.Agent import Agent
from utils.replayMemory import replayMemory
from utils.util import Color
from utils.util import soft_update


class DQN(Agent):
    def __init__(self, env, Q_net: Type[torch.nn.Module], config):
        super(DQN, self).__init__(env, config)
        self._Q = Q_net  # constructor for Q network
        self.Q = self._Q()
        if hasattr(self.Q, 'dueling'):
            self.dueling = True
            print(f'{Color.INFO}using the dueling network{Color.END}')
        self.target_Q = copy.deepcopy(self.Q)
        self.optimizer = optim.Adam(self.Q.parameters(),
                                    lr=self.config.get('learning_rate', 0.01),
                                    eps=1e-4)
        self.replayMemory = replayMemory(self.config.get('memory_capacity', 20000), self.config.get('batch_size', 256))

        self._epsilon = None  # decay according to episode number during training

    def run_reset(self):
        super(DQN, self).run_reset()
        self.replayMemory.reset()
        self.Q = self._Q()
        self.target_Q = copy.deepcopy(self.Q)
        self.optimizer = optim.Adam(self.Q.parameters(),
                                    lr=self.config.get('learning_rate', 0.01),
                                    eps=1e-4)

    def episode_reset(self):
        super(DQN, self).episode_reset()
        self.update_epsilon()

    def run_an_episode(self):
        while not self.done:
            self.length[self._run][self._episode] += 1
            self.select_action()
            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            self.rewards[self._run][self._episode] += self.reward

            # save experience
            experience = (self.state, self.action, self.reward, self.next_state, self.done)
            self.replayMemory.add(experience)

            # only start to learn when there are enough experiences to sample from
            if self.replayMemory.ready:
                self.learn()

            self.state = self.next_state
        print(f'\r{format(self._episode + 1, ">3")}th episode: '
              f'{format(self.length[self._run][self._episode], ">3")} steps, '
              f'rewards: {format(self.rewards[self._run][self._episode], ">5.1f")}, '
              f'running reward: {format(self._running_reward, ">7.3f")}, '
              f'learning rate: {format(self._learning_rate, ">5")}, '
              f'epsilon: {format(self._epsilon, ".4f")}', end='')

    def select_action(self):
        """select action according to `self.Q` given current state"""
        # todo: check state dim
        state = torch.tensor(self.state).float().unsqueeze(0)
        if random.random() > self._epsilon:
            # some modules behave differently in training/evaluation mode, e.g. Dropout, BatchNorm
            self.Q.eval()
            with torch.no_grad():
                actions_value = self.Q(state)
            self.Q.train()

            self.action = actions_value.argmax().item()
            value_to_show = [float(format(v, '.10f')) for v in actions_value.tolist()[0]]
            self.logger.info(f'actions_value: {value_to_show}, choose action: {self.action}')
        else:
            self.action = self.env.action_space.sample()
            self.logger.info(f'choose randomly, action: {self.action}')

    def learn(self):
        self._states, self._actions, self._rewards, self._next_states, self._dones = self.replayMemory.sample()
        loss = F.mse_loss(self.current_states_value, self.target_value)
        self.logger.info(f'loss: {loss.item()}')
        self.perform_gradient_descent(loss)
        self.update_target_Q()

    def perform_gradient_descent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad = self.config.get('clip_grad', None)
        if clip_grad: torch.nn.utils.clip_grad_norm_(self.Q.parameters(), clip_grad)
        self.optimizer.step()

    def update_target_Q(self):
        """update target Q network if exist"""
        if self.length[self._run].sum() % self.config.get('Q_update_interval', 0) == 0:
            if 'tau' in self.config:
                soft_update(self.Q, self.target_Q, self.config.get('tau'))
            else:
                self.target_Q = copy.deepcopy(self.Q)

    def save_policy(self):
        """save the parameter of the Q network(`self.Q) when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run}_{self._episode + 1}.pt'
            torch.save(self.Q.state_dict(), os.path.join(self.policy_path, name))

    def update_epsilon(self):
        """get the probability of picking action randomly
        epsilon should decay as more episodes have been seen and higher rewards we get"""
        # todo: epsilon decay
        epsilon0 = self.config['epsilon']
        min_epsilon = self.config.get('min_epsilon', 0.01)
        if self._episode == 0:
            self._epsilon = epsilon0
        elif self.running_rewards[self._run][self._episode - 1] >= self.goal:
            self._epsilon = min_epsilon
        else:
            self._epsilon *= self.config.get('epsilon_decay_rate', 0.99)
            self._epsilon = max(self._epsilon, min_epsilon)

    @property
    def current_states_value(self):
        return self.Q(self._states).gather(1, self._actions.long())

    @property
    def next_states_value(self):
        return self.target_Q(self._next_states).detach().max(1)[0].unsqueeze(1)

    @property
    def target_value(self):
        return self._rewards + self.config.get('discount_factor', 0.99) * self.next_states_value * (1 - self._dones)

    def _evaluate_policy(self, file_name, episodes):
        # set up Q network for test
        file = os.path.join(self.policy_path, file_name)
        self.Q = self._Q()
        self.Q.load_state_dict(torch.load(file))
        self.Q.eval()

        # define some variable to record performance
        rewards = np.zeros(episodes)
        running_rewards = np.zeros(episodes)

        for episode in range(episodes):
            # test for an episode
            self.env.seed()
            state = self.env.reset()
            done = False
            while not done:
                # self.env.render()
                # time.sleep(0.01)
                state = torch.tensor(state).float().unsqueeze(0)
                with torch.no_grad(): actions_value = self.Q(state)
                action = actions_value.argmax().item()
                next_state, reward, done, _ = self.env.step(action)
                rewards[episode] += reward
                state = next_state

            running_rewards[episode] = np.mean(rewards[max(episode - self.window + 1, 0):episode + 1])
            print(f'\rTesting policy {file_name}: episode: {episode + 1}, '
                  f'reward: {rewards[episode]}, running reward: {format(running_rewards[episode], ".2f")}', end=' ')

        # running rewards only make sense when the agent runs at least `self.window` episodes
        if np.any(running_rewards[self.window - 1:] >= self.goal):
            print(f'{Color.SUCCESS}Test Passed{Color.END}')
        else:
            print(f'{Color.FAIL}Test Failed{Color.END}')
            os.remove(file)

        # plot the test result of this policy
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('rewards')
        name = f'{os.path.splitext(file_name)[0]} test results'  # get filename without extension
        ax.set_title(name)
        ax.plot(np.arange(1, episodes + 1), rewards, label='test')
        ax.plot(np.arange(1, episodes + 1), running_rewards, label='running rewards')
        ax.hlines(y=self.goal, xmin=1, xmax=self.episode_num + 1, label='goal', alpha=0.5)
        ax.legend(loc='upper left')
        plt.savefig(os.path.join(self.results_path, name))
        fig.clear()
        # todo: another way to handle multiple figures
        plt.close('all')
