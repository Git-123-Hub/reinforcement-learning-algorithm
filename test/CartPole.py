############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
############################################

import gym
import torch.nn as nn
from torch.distributions import Categorical

from policy_based.REINFORCE import REINFORCE
from utils.const import get_base_config
from utils.util import compare
from value_based.DDQN import DDQN
from value_based.DDQN_PER import DDQN_PER
from value_based.DQN import DQN
from value_based.DuelingDQN import DuelingQNet


class QNet(nn.Module):
    """
    input state, output an array of length action space
    """

    def __init__(self):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 2),
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        # input state size, output probability on each action
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.Dropout(p=0.6),
            nn.Linear(128, 2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        probs = self.fc(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


if __name__ == '__main__':
    config = get_base_config()
    config['results'] = './CartPole_results'
    config['policy'] = './CartPole_policy'
    config['seed'] = 7511267
    config['run_num'] = 5
    config['episode_num'] = 1000
    config['min_epsilon'] = 0.001
    config['Q_update_interval'] = 20
    config['tau'] = 0.3

    env = gym.make('CartPole-v1')
    # agents = [DDQN]
    # # agents = [DDQN, DDQN_PER]
    # for agent in agents:
    #     agent(env, QNet, config).train()
    #     agent(env, QNet, config).test(10)
    # compare([agent.__name__ for agent in agents], config['results'])

    config['learning_rate_decay_rate'] = 1
    agent = REINFORCE(env, Policy, config)
    # agent.train()
    agent.test(10)
