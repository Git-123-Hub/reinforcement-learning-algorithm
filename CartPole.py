############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
############################################

import gym
import torch.nn as nn

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
    'run_num': 1,
    'episode_num': 50,
    'learning_rate': 0.01,
    'discount_factor': 0.99,
    'epsilon': [1, 0.01],
    "epsilon_decay_rate_denominator": 1,
    # "clip_grad": 0.7
    # for nature_DQN
    'Q_update_interval': 4,
    # for DDQN
    # tau*Q.parameter will be copied to target_Q,
    # considering the Q is `learning`, so a bigger tau might make the algorithm more stable
    'tau': 0.7,
}

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQN(env, QNet, config)
    agent.train()
