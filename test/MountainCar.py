############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem MountainCart-v0
############################################


import gym
import torch.nn as nn
from torch.distributions import Categorical

from policy_based import REINFORCE
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
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            # nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        probs = self.fc(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class ModifyReward(gym.Wrapper):
    def __init__(self, env):
        super(ModifyReward, self).__init__(env)
        self.env = env
        self.env.spec.reward_threshold = -50
        # note that this `new goal` is empirical,
        # only to make sure that once the agent reaches this goal, it can really learn a good policy

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # if the agent reach the goal, return more reward
        if next_state[0] >= self.env.goal_position:  # more reward if the agent reaches the goal
            reward += 50
        elif next_state[0] >= 0.4:  # encourage the agent go further
            reward += 3

        return next_state, reward, done, info


if __name__ == '__main__':
    env = ModifyReward(gym.make('MountainCar-v0'))
    # env = gym.make('MountainCar-v0')

    config = get_base_config()
    config['results'] = './Mountain_results'
    config['policy'] = './Mountain_policy'

    config['seed'] = 74512569
    config['run_num'] = 3
    config['episode_num'] = 500

    config['memory_capacity'] = 10000
    config['batch_size'] = 64

    config['learning_rate'] = 1e-3
    config['learning_rate_decay_rate'] = 1

    # value based agent
    config['Q_update_interval'] = 4
    config['tau'] = 0.1

    config['epsilon'] = 1  # start epsilon
    config['epsilon_decay_rate'] = 0.98
    config['min_epsilon'] = 0.01

    agent = DDQN(env, QNet, config)
    agent.train()
    # agent.test(20, render=False)

    # agent = DDQN_PER(env, QNet, config)
    # agent.train()

    # policy based
    # a = REINFORCE(env, Policy, config)
    # a.train()
