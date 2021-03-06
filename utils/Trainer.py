############################################
# @Author: Git-123-Hub
# @Date: 2021/11/18
# @Description: Trainer to train the agent for multiple run
############################################
import os

import numpy as np
from policy_based import A3C
from utils import Agent
import matplotlib.pyplot as plt
from utils.const import Color
from utils.util import initial_folder


class Trainer:
    def __init__(self, agent: A3C, run: int):
        self.agent = agent
        self.run_num = run
        self.rewards = np.zeros((run, agent.episode_num), dtype=float)
        self.running_rewards = np.zeros((run, agent.episode_num), dtype=float)

    def train(self):
        for run in range(self.run_num):
            print(f'train {self.agent.__class__.__name__} to solve {self.agent.env_id} for the {run + 1}th run:')
            # todo: change run_reset() to reset()
            self.agent.reset()
            self.agent.run = run + 1
            self.agent.train()

            # copy training result to `self.rewards` for statistical analyze
            self.rewards[run][:] = self.agent.rewards[:]
            self.running_rewards[run][:] = self.agent.running_rewards[:]

        # plot all the training result
        # initialize a figure
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('running reward')

        # prepare data
        x = np.arange(1, self.agent.episode_num + 1)
        mean = np.mean(self.running_rewards, axis=0)
        std = np.std(self.running_rewards, axis=0)
        color = getattr(Color, self.agent.__class__.__name__)  # different color for different agent

        ax.plot(x, mean, color=color)
        ax.plot(x, mean - std, color=color, alpha=0.2)
        ax.plot(x, mean + std, color=color, alpha=0.2)
        ax.fill_between(x, y1=mean - std, y2=mean + std, color=color, alpha=0.1)
        ax.hlines(y=self.agent.goal, xmin=1, xmax=self.agent.episode_num, label='goal', color=Color.GOAL)

        # for run in range(self.run_num):  # plot running reward of each run separately
        #     ax.plot(x, self.running_rewards[run], label=f'{run + 1}th run')
        # ax.legend(loc='lower right')

        name = f'running reward of {self.agent.__class__.__name__} solving {self.agent.env_id}'
        ax.set_title(name)
        plt.savefig(os.path.join(self.agent.result_path, name))
        plt.close(fig)
