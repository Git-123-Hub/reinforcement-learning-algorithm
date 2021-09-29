############################################
# @Author: Git-123-Hub
# @Date: 2021/9/28
# @Description: test the performance of RL algorithm on problem HalfCheetah-v3
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
        self.max_action = max_action
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state) * self.max_action


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).float().unsqueeze(0)
        x = torch.cat([state, action], 1)
        return self.fc(x)


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v3')
    config = get_base_config()
    config['results'] = './HalfCheetah_results'
    config['policy'] = './HalfCheetah_policy'
    config['seed'] = 482307631
    config['run_num'] = 1
    config['episode_num'] = 500

    config['memory_capacity'] = 1e5
    config['batch_size'] = 256

    config['Q_update_interval'] = 1
    config['tau'] = 0.005

    config['learning_rate'] = 3e-4
    config['learning_rate_decay_rate'] = 1

    # agent = DDPG(env, Actor, Critic, config)
    # agent.train()
    # agent.test()

    config['update_interval'] = 2
    config['noise_std'] = 1
    config['noise_clip'] = 0.5
    config['noise_factor'] = 0.2
    agent = TD3(env, Actor, Critic, config)
    agent.train()
    # agent.test(render=True)
