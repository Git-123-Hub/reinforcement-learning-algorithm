############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem Acrobot-v1
############################################


import gym
import torch.nn as nn

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
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.fc(x)


config = {
    'memory_capacity': 20000,
    'batch_size': 256,
    'alpha': 0.3,
    'beta': 0.3,
    'seed': 75617127,
    'run_num': 5,
    'episode_num': 700,
    'learning_rate': 0.01,
    'clear_result': False,
    'clear_policy': False,
    'discount_factor': 0.99,
    'epsilon': [1, 0.01],
    'tau': 0.2,
    'Q_update_interval': 10,  # if not specified, update every step, i.e. equals 0
}

if __name__ == '__main__':
    config['results'] = './Acrobot-results'
    config['policy'] = './Acrobot-policy'
    env = gym.make('Acrobot-v1')
    # agents = [DQN, DDQN]
    agents = [DDQN, DDQN_PER]
    for agent in agents:
        # agent(env, QNet, config).train()
        agent(env, QNet, config).test(5)
    # compare([agent.__name__ for agent in agents], './Acrobot-results')
