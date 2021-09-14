############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
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
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.fc(x)


config = {
    'memory_capacity': 20000,
    'batch_size': 256,
    'alpha': 0.5,
    'beta': 0.5,
    # 'seed': 756711267,
    'run_num': 1,
    'episode_num': 300,
    'learning_rate': 0.01,
    'clear_result': False,
    'clear_policy': False,
    'discount_factor': 0.99,
    'epsilon': [1, 0.01],
    # soft update parameter, if not specified, deepcopy Q to target_Q
    'tau': 0.3,
    # "clip_grad": 0.7,
    # parameters for NatureDQN
    'Q_update_interval': 50,  # if not specified, update every step, i.e. equals 0
}

if __name__ == '__main__':
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
