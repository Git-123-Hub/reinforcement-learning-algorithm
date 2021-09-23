############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: solve problem Pendulum using rl algorithm
############################################
import gym
import numpy as np
import torch
from torch import nn

from policy_based.DDPG import DDPG
from utils.const import get_base_config


class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)
        return self.fc(state)


class Critic(nn.Module):
    # todo: how to handle state-action value: input state, or input state+action
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float().unsqueeze(0)

        if isinstance(action, np.ndarray):
            action = torch.tensor(action).float().unsqueeze(0)

        # todo: check data dimension, this method might be wrong
        return self.fc(torch.cat([state, action], 1))


# wrap the env with a specific goal
class ModifyReward(gym.Wrapper):
    def __init__(self, env):
        super(ModifyReward, self).__init__(env)
        self.env = env
        self.env.spec.reward_threshold = -100


if __name__ == '__main__':
    env = ModifyReward(gym.make('Pendulum-v0'))
    config = get_base_config()

    agent = DDPG(env, Actor, Critic, config)
    agent.train()
