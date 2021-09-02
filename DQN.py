############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: implementation of Deep-Q-Learning
############################################
import random
from typing import Type

import torch
from torch import optim
import torch.nn.functional as F
from Agent import Agent
from const import DEFAULT
from replayMemory import replayMemory


class DQN(Agent):
    def __init__(self, env, Q_net: Type[torch.nn.Module], config):
        super(DQN, self).__init__(env, config)
        self._Q = Q_net  # constructor for Q network
        self.Q = self._Q()
        self.optimizer = optim.Adam(self.Q.parameters(),
                                    lr=self.config.get('learning_rate', DEFAULT['learning_rate']),
                                    eps=1e-4)
        self.replayMemory: replayMemory = replayMemory(**self.config['replay_config'])

    def run_reset(self):
        super(DQN, self).run_reset()
        self.replayMemory.reset()
        self.Q = self._Q()
        self.optimizer = optim.Adam(self.Q.parameters(),
                                    lr=self.config.get('learning_rate', DEFAULT['learning_rate']),
                                    eps=1e-4)

    def run_an_episode(self):
        while not self.done:
            self.length[self._run][self._episode] += 1
            self.select_action()
            self.execute_action()
            self.save_experience()
            self.learn()
            self.state = self.next_state
        print(f'\r{format(self._episode + 1, ">3")}th episode: '
              f'{format(self.length[self._run][self._episode], ">3")} steps, '
              f'rewards: {format(self.rewards[self._run][self._episode], ">5.1f")}, '
              f'running reward: {format(self._running_reward, ">7.3f")}, '
              f'learning rate: {format(self._learning_rate, ">5")}, '
              f'epsilon: {format(self.epsilon, ".4f")}', end='')

    def select_action(self):
        """select action according to `self.Q` given current state"""
        # todo: check state dim
        state = torch.tensor(self.state).float().unsqueeze(0)
        if random.random() > self.epsilon:
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

    def execute_action(self):
        self.next_state, self.reward, self.done, _ = self.env.step(self.action)
        self.rewards[self._run][self._episode] += self.reward

    def save_experience(self):
        """save the `experience` to `self.replayMemory`"""
        experience = (self.state, self.action, self.reward, self.next_state, self.done)
        self.replayMemory.add(experience)

    def learn(self):
        if self.replayMemory.ready:
            experiences = self.replayMemory.sample()
            loss = self.compute_loss(*experiences)
            self.perform_gradient_descent(loss)

    @property
    def epsilon(self, episode_num=None):
        """get the probability of picking action randomly
        epsilon should decay as more episodes have been seen and higher rewards we get"""
        # todo: epsilon decay
        # return self.config.get('epsilon', DEFAULT['epsilon'])
        if episode_num is None: episode_num = self._episode
        ep_range = self.config['epsilon']
        return ep_range[0] / (1 + episode_num / self.config["epsilon_decay_rate_denominator"])

    def compute_loss(self, states, actions, rewards, next_states, dones):
        """compute the loss the train the {self.Q}"""
        # calculate q_values on given {states} and given {actions} using self.Q
        q_values = self.Q(states).gather(1, actions.long())
        with torch.no_grad():
            # calculate q_targets given {next_states} and take the maximum action using self.targetQ
            q_targets_next_state = self.Q(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + self.config['discount_factor'] * q_targets_next_state * (1 - dones)
        loss = F.mse_loss(q_values, q_targets)
        self.logger.info(f'Loss: {loss.item()}')
        return loss

    def perform_gradient_descent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad = self.config.get('clip_grad', DEFAULT['clip_grad'])
        if clip_grad: torch.nn.utils.clip_grad_norm_(self.Q.parameters(), clip_grad)
        self.optimizer.step()

