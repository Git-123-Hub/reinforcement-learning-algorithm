############################################
# @Author: Git-123-Hub
# @Date: 2021/9/7
# @Description: implementation of Dueling DQN
############################################
from torch import nn


# Note that the main innovation of Dueling DQN lies in the structure of the network
# so here i only define the net, you can pass it to other algorithm


class DuelingQNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super(DuelingQNet, self).__init__()
        self.dueling = True
        self.base_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, state):
        base_output = self.base_layer(state)
        value = self.value_stream(base_output)
        advantage = self.advantage_stream(base_output)
        return value + (advantage - advantage.mean())
