############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem MountainCart-v0
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


if __name__ == '__main__':
    config = get_base_config()
    config['results'] = './Mountain_results'
    config['policy'] = './Mountain_policy'
    # config['seed'] = 74210479
    config['run_number'] = 5
    config['episode_num'] = 1000
    config['learning_rate'] = 0.03
    config['learning_rate_decay_rate'] = 0.95
    config['Q_update_interval'] = 30
    # config['tau'] = 0.3
    # config['epsilon_decay_rate'] = 0.9999

    env = gym.make('MountainCar-v0')
    agents = [DDQN, DDQN_PER]
    for agent in agents:
        agent(env, QNet, config).train()
        # agent(env, QNet, config).test(30)
    compare([agent.__name__ for agent in agents], './Mountain_results')
