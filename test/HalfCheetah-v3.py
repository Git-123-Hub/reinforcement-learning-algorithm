############################################
# @Author: Git-123-Hub
# @Date: 2021/9/28
# @Description: test the performance of RL algorithm on problem HalfCheetah-v3
############################################
import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from policy_based import DDPG, TD3, SAC
from utils.const import get_base_config
from utils.util import setup_logger

logger = setup_logger('debug.log', 'debug')

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


class StochasticActor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(StochasticActor, self).__init__()
        self.base_fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_output = nn.Linear(256, action_dim)
        self.log_std_output = nn.Linear(256, action_dim)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        base = self.base_fc(state)
        # todo: check data dimension
        mean, log_std = self.mean_output(base).squeeze(0), self.log_std_output(base).squeeze(0)
        # todo: maybe use min_log and max_log to clamp value
        log_std.clip_(-20, 2)
        # NOTE that the clip value of log_std is taken from rlKit
        # (https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/policies/gaussian_policy.py)
        std = torch.exp(log_std)
        logger.info(f'mean: {mean}, std: {std}')
        return Normal(mean, std)


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v3')
    config = get_base_config()
    config['results'] = './HalfCheetah_results'
    config['policy'] = './HalfCheetah_policy'
    # config['seed'] = 482307631
    config['run_num'] = 5
    config['episode_num'] = 1000

    config['memory_capacity'] = 1e5
    config['batch_size'] = 256

    config['Q_update_interval'] = 1
    config['tau'] = 0.005

    config['learning_rate'] = 3e-4

    # agent = DDPG(env, Actor, Critic, config)
    # agent.train()
    # agent.test()

    config['update_interval'] = 2
    config['noise_std'] = 1
    config['noise_clip'] = 0.5
    config['noise_factor'] = 0.2
    # agent = TD3(env, Actor, Critic, config)
    # agent.train()
    # agent.test(render=True)

    config['learning_rate'] = 1e-3
    agent = SAC(env, StochasticActor, Critic, config)
    agent.train()
