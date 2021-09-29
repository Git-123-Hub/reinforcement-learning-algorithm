############################################
# @Author: Git-123-Hub
# @Date: 2021/9/6
# @Description: implementation of Double Deep Q Learning
############################################
from typing import Type

import torch

from value_based import DQN


class DDQN(DQN):
    def __init__(self, env, Q_net: Type[torch.nn.Module], config):
        super(DDQN, self).__init__(env, Q_net, config)

    # DDQN differs from DQN on how to calculate next_states_value
    @property
    def next_states_value(self):
        # find the best action with max Q value using `self.Q`
        best_action = self.Q(self._next_states).detach().argmax(1)
        # calculate next_states_value using `self.target_Q` and the best action
        return self.target_Q(self._next_states).detach().gather(1, best_action.unsqueeze(1))
