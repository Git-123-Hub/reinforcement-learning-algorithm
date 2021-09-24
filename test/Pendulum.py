############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: solve problem Pendulum using rl algorithm
############################################
import gym
import numpy as np
import torch
from torch import nn

from policy_based.DDPG import DDPG
from utils.const import get_base_config


class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


class Critic(nn.Module):
    # todo: how to handle state-action value: input state, or input state+action
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).float().unsqueeze(0)
        return self.fc(torch.cat([state, action], 1))


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    config = get_base_config()
    config['results'] = './Pendulum_results'
    config['policy'] = './Pendulum_policy'
    config['seed'] = 13628075
    config['run_num'] = 5
    config['episode_num'] = 1000

    config['Q_update_interval'] = 1
    config['tau'] = 0.05

    config['learning_rate'] = 0.001
    config['learning_rate_decay_rate'] = 1

    agent = DDPG(env, Actor, Critic, config)
    agent.train()
    # agent.test()
