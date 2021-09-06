############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
############################################

import gym
import torch.nn as nn

from DDQN import DDQN
from DQN import DQN


class QNet(nn.Module):
    """
    input state, output an array of length action space
    """

    def __init__(self):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.fc(x)


config = {
    # todo use schema to validate the structure of the config
    'replay_config': {
        'capacity': 40000,
        'batch_size': 256,
    },
    'seed': 123322433,
    'run_num': 2,
    'episode_num': 200,
    'learning_rate': 0.01,
    'clear_result': False,
    'clear_policy': False,
    'discount_factor': 0.99,
    'epsilon': [1, 0.01],
    "epsilon_decay_rate_denominator": 1,
    # "clip_grad": 0.7
    # parameters for NatureDQN
    'Q_update_interval': 20,  # if not specified, update every step, i.e. equals 0
    # for DDQN
    # tau*Q.parameter will be copied to target_Q,
    # considering the Q is `learning`, so a bigger tau might make the algorithm more stable
    # 'tau': 0.01,  # if not specified, deepcopy Q to target_Q
}

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # test DQN
    agent = DDQN(env, QNet, config)
    agent.train()
    # agent.test(5)
