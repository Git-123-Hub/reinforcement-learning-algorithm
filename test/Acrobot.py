############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem Acrobot-v1
############################################


import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from policy_based.ActorCritic import ActorCritic
from policy_based.REINFORCE import REINFORCE
from utils.const import get_base_config
from utils.util import compare
from value_based import DDQN, DDQN_PER, DQN, DuelingQNet


class QNet(nn.Module):
    """
    input state, output an array of length action space
    """

    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        # input state size, output probability on each action
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        probs = self.fc(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    config = get_base_config()

    # base config
    config['results'] = './Acrobot-results'
    config['policy'] = './Acrobot-policy'
    config['seed'] = 756170127
    config['run_num'] = 3
    config['episode_num'] = 1000

    config['learning_rate'] = 0.001
    config['learning_rate_decay_rate'] = 1

    # config for DQN
    config['min_epsilon'] = 0.001
    config['Q_update_interval'] = 20
    config['tau'] = 0.1

    agent = DDQN(env, QNet, config)
    agent.train()

    agent = DDQN_PER(env, QNet, config)
    agent.train()

    config['learning_rate'] = 0.005
    agent = REINFORCE(env, Policy, config)
    agent.train()
    # agent.test()

    agent = ActorCritic(env, Policy, Critic, config)
    agent.train()
    # agent.test()

    compare(['DDQN', 'DDQN_PER', 'REINFORCE', 'ActorCritic'], './Acrobot-results')
