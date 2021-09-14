############################################
# @Author: Git-123-Hub
# @Date: 2021/9/5
# @Description: test RL agent's performance on solving problem LunarLander
############################################
import gym
from torch import nn

from utils.const import get_base_config
from utils.util import compare
from value_based.DDQN import DDQN
from value_based.DDQN_PER import DDQN_PER
from value_based.DQN import DQN


class QNet(nn.Module):
    """
    input state, output an array of length action space
    """

    def __init__(self):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    config = get_base_config()
    config['seed'] = 123782433
    config['Q_update_interval'] = 20
    config['tau'] = 0.3

    config['results'] = './LunarLander-results'
    config['policy'] = './LunarLander-policy'
    env = gym.make('LunarLander-v2')
    agent = DQN(env, QNet, config)
    agent.train()
    # agent.test(5)
