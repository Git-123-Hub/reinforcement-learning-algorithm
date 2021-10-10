############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem MountainCart-v0
############################################


import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from policy_based import REINFORCE, REINFORCE_BASELINE
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
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        probs = self.fc(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


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
        if next_state[0] >= self.env.goal_position:  # more reward if the agent reaches the goal(0.5)
            reward += 10
        elif next_state[0] >= 0.4:  # encourage the agent go further
            reward += 10
        elif next_state[0] >= 0.3:
            reward += 5
        elif next_state[0] >= 0.2:
            reward += 3
        elif next_state[0] >= 0.1:
            reward += 1

        return next_state, reward, done, info


if __name__ == '__main__':
    # NOTE that the performance of solving this problem is not stable
    env = ModifyReward(gym.make('MountainCar-v0'))
    # env = gym.make('MountainCar-v0')

    config = get_base_config()
    config['results'] = './Mountain_results'
    config['policy'] = './Mountain_policy'

    config['seed'] = 4357436
    config['run_num'] = 3
    config['episode_num'] = 1000

    config['memory_capacity'] = 10000
    config['batch_size'] = 256

    config['learning_rate'] = 5e-4

    # value based agent
    config['Q_update_interval'] = 5
    config['tau'] = 0.01

    config['epsilon_decay_rate'] = 0.992
    config['min_epsilon'] = 0.01

    agent = DQN(env, QNet, config)
    agent.train()

    agent = DDQN(env, QNet, config)
    agent.train()

    env = ModifyReward(gym.make('MountainCar-v0'))
    agent = DDQN_PER(env, QNet, config)
    agent.train()

    # policy based
    a = REINFORCE(env, Policy, config)
    a.train()

    agent = REINFORCE_BASELINE(env, Policy, Critic, config)
    agent.train()

    compare(['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'REINFORCE_BASELINE'], config['results'])
