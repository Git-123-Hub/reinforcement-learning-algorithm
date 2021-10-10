############################################
# @Author: Git-123-Hub
# @Date: 2021/9/5
# @Description: test RL agent's performance on solving problem LunarLander
############################################
import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from policy_based import REINFORCE, REINFORCE_BASELINE
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
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
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
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    config = get_base_config()
    config['results'] = './LunarLander-results'
    config['policy'] = './LunarLander-policy'

    config['seed'] = 4375233
    config['run_num'] = 3
    config['episode_num'] = 400

    # value based agent
    config['Q_update_interval'] = 4
    config['tau'] = 0.01
    config['learning_rate'] = 1e-4

    agent = DQN(env, QNet, config)
    agent.train()

    agent = DDQN(env, QNet, config)
    agent.train()

    agent = DDQN_PER(env, QNet, config)
    agent.train()

    # policy based
    # NOTE that these method can't solve the problem, it can learn something, but can't reach the goal
    config['learning_rate'] = 5e-3
    config['episode_num'] = 1000
    a = REINFORCE(env, Policy, config)
    a.train()

    agent = REINFORCE_BASELINE(env, Policy, Critic, config)
    agent.train()

    compare(['DQN', 'DDQN'], config['results'])
    compare(['DQN', 'DDQN', 'DDQN_PER'], config['results'])
    compare(['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'REINFORCE_BASELINE'], config['results'])
