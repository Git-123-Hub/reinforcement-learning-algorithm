############################################
# @Author: Git-123-Hub
# @Date: 2021/9/5
# @Description: test RL agent's performance on solving problem LunarLander
############################################
import gym
from torch import nn
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


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    config = get_base_config()
    config['results'] = './LunarLander-results'
    config['policy'] = './LunarLander-policy'

    config['seed'] = 4375233
    config['run_num'] = 10
    config['episode_num'] = 600

    # value based agent
    config['Q_update_interval'] = 4
    config['tau'] = 0.01  # or maybe 1e-3
    config['learning_rate_decay_rate'] = 1
    config['learning_rate'] = 1e-4
    # config['min_learning_rate'] = 1e-5
    agent = DDQN(env, DuelingQNet, config)
    agent.train()
    agent.test(render=False)
