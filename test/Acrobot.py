############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem Acrobot-v1
############################################


import gym
import torch.nn as nn

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


if __name__ == '__main__':
    config = get_base_config()
    config['seed'] = 75617127
    config['episode_num'] = 700
    config['results'] = './Acrobot-results'
    config['policy'] = './Acrobot-policy'
    env = gym.make('Acrobot-v1')
    # agents = [DQN, DDQN]
    agents = [DDQN, DDQN_PER]
    for agent in agents:
        # agent(env, QNet, config).train()
        agent(env, QNet, config).test(5)
    # compare([agent.__name__ for agent in agents], './Acrobot-results')
