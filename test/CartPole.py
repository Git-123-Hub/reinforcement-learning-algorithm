############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
############################################

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from policy_based import REINFORCE
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
            nn.Linear(state_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        # input state size, output probability on each action
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Dropout(p=0.6),
            nn.Linear(128, action_dim),
            nn.ReLU(),
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
            nn.Linear(state_dim, 128),
            # nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.ReLU(),
            # nn.Linear(64, 1),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    config = get_base_config()

    # base config
    config['results'] = './CartPole_results'
    config['policy'] = './CartPole_policy'
    config['seed'] = 7511267
    config['run_num'] = 5
    config['episode_num'] = 1000
    config['min_epsilon'] = 0.001
    config['Q_update_interval'] = 20
    config['tau'] = 0.3

    config['learning_rate'] = 0.001
    config['learning_rate_decay_rate'] = 1

    # config for DQN
    # config['min_epsilon'] = 0.001
    # config['Q_update_interval'] = 20
    # config['tau'] = 0.3
    # agents = [DDQN]
    # # agents = [DDQN, DDQN_PER]
    # for agent in agents:
    #     agent(env, QNet, config).train()
    #     agent(env, QNet, config).test(10)
    # compare([agent.__name__ for agent in agents], config['results'])

    agent = REINFORCE(env, Policy, config)
    agent.train()
    # agent.test()

    agent = ActorCritic(env, Policy, Critic, config)
    agent.train()
    # agent.test()
