############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: solve problem Pendulum using rl algorithm
############################################
import gym
import numpy as np
import torch
from torch import nn

from policy_based import DDPG, TD3
from utils.const import get_base_config


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).float().unsqueeze(0)
        return self.fc(torch.cat([state, action], 1))


if __name__ == '__main__':
    # NOTE that this is no goal for Pendulum-v0, but as you can see in the result, the agent did learn something
    env = gym.make('Pendulum-v0')
    config = get_base_config()
    config['results'] = './Pendulum_results'
    config['policy'] = './Pendulum_policy'
    config['seed'] = 103423575
    config['run_num'] = 3
    config['episode_num'] = 300

    config['memory_capacity'] = 10000
    config['batch_size'] = 64

    config['Q_update_interval'] = 1
    config['tau'] = 0.01

    config['learning_rate'] = 0.0005

    agent = DDPG(env, Actor, Critic, config)
    agent.train()
    # agent.test()

    config['update_interval'] = 2
    config['noise_std'] = 1
    config['noise_clip'] = 0.5
    config['noise_factor'] = 0.2

    agent = TD3(env, Actor, Critic, config)
    agent.train()
    # agent.test()
