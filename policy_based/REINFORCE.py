############################################
# @Author: Git-123-Hub
# @Date: 2021/9/17
# @Description: implementation of REINFORCE(Monte-Carlo Policy gradient)
############################################
import os

import gym
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt

from utils.Agent import Agent
from utils.const import Color


class REINFORCE(Agent):
    def __init__(self, env, policyNet, config):
        super(REINFORCE, self).__init__(env, config)
        self._policyNet = policyNet
        self.policy = self._policyNet()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get('learning_rate', 0.01))

    def run_reset(self):
        super(REINFORCE, self).run_reset()
        self.policy = self._policyNet()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get('learning_rate', 0.01))

    def run_an_episode(self):
        episode_log_prob, episode_reward = [], []
        while not self.done:
            self.length[self._run][self._episode] += 1

            # select action
            state = torch.tensor(self.state).float().unsqueeze(0)
            self.action, log_prob = self.policy(state)
            episode_log_prob.append(log_prob)

            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            # self.env.render()
            episode_reward.append(self.reward)
            self.rewards[self._run][self._episode] += self.reward
            self.state = self.next_state

        self.learn(episode_reward, episode_log_prob)  # learn when an episode finishes

        print(f'\r{format(self._episode + 1, ">4")}th episode: '
              f'{format(self.length[self._run][self._episode], ">3")} steps, '
              f'rewards: {format(self.rewards[self._run][self._episode], ">7.2f")}, '
              f'running reward: {format(self._running_reward, ">7.2f")}, '
              f'learning rate: {format(self._learning_rate, ">7.6f")}, ', end='')

    def learn(self, reward_list, log_prob_list):
        returns = np.zeros_like(reward_list)
        eps = np.finfo(np.float32).eps.item()  # tiny non-negative number
        R = 0
        for index in reversed(range(len(reward_list))):
            R = reward_list[index] + self.config.get('discount_factor', 0.99) * R
            returns[index] = R
        returns = (returns - returns.mean()) / (returns.std() + eps)
        loss_list = torch.cat(log_prob_list) * torch.from_numpy(-returns)
        # Note the negative sign for `returns` to change gradient ascent to gradient decent

        loss = loss_list.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # todo: evaluate policy
    def save_policy(self):
        """save the parameter of the policy network(`self.policy`) when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.policy.state_dict(), os.path.join(self.policy_path, name))

    def _evaluate_policy(self, file_name, episodes):
        # set up policy network for test
        file = os.path.join(self.policy_path, file_name)
        self.policy.load_state_dict(torch.load(file))
        self.policy.eval()

        # define some variable to record performance
        rewards = np.zeros(episodes)
        running_rewards = np.zeros(episodes)
        # remake the environment just in case of the training environment is modified
        self.env = gym.make(self.env_id)
        for episode in range(episodes):
            # test for an episode
            self.env.seed()
            state = self.env.reset()
            done = False
            while not done:
                # self.env.render()
                # time.sleep(0.01)
                state = torch.tensor(state).float().unsqueeze(0)
                with torch.no_grad(): action, _ = self.policy(state)
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
            # consider there might be a lot of policies saved
            # you can delete a policy if it fails the test
            # or you can comment the line below to not to delete it
            os.remove(file)

        # plot the test result of this policy
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('rewards')
        name = f'{os.path.splitext(file_name)[0]} test results'  # get filename without extension
        ax.set_title(name)
        ax.plot(np.arange(1, episodes + 1), rewards, label='test')
        ax.plot(np.arange(1, episodes + 1), running_rewards, label='running rewards')
        ax.hlines(y=self.goal, xmin=1, xmax=episodes, label='goal', alpha=0.5)
        ax.legend(loc='upper left')
        plt.savefig(os.path.join(self.results_path, name))
        fig.clear()
        # todo: another way to handle multiple figures
        plt.close('all')
