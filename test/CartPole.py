############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
############################################

import gym
import torch.nn as nn

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
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    config = get_base_config()
    config['seed'] = 75071267
    config['Q_update_interval'] = 10

    env = gym.make('CartPole-v1')
    # agents = [DQN, DDQN]
    agents = [DDQN_PER]
    for agent in agents:
        agent(env, QNet, config).train()
    # compare([agent.__name__ for agent in agents])

    # config['results'] = './result2'
    # config['policy'] = './policy2'
    # for agent in agents:
    #     agent(env, DuelingQNet, config).train()
