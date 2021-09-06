############################################
# @Author: Git-123-Hub
# @Date: 2021/9/6
# @Description: implementation of Deep-Q-Learning(Nature version)
############################################
import copy
from typing import Type

import torch

from DQN import DQN


#########################################
# improvement of NatureDQN over DQN
# a) add target network(target_Q) and use it to calculate next states' Q values
# b) update target network every few steps
#########################################


class NatureDQN(DQN):
    def __init__(self, env, Q_net: Type[torch.nn.Module], config):
        super().__init__(env, Q_net, config)
        self.target_Q = copy.deepcopy(self.Q)

    def run_reset(self):
        super(NatureDQN, self).run_reset()
        self.target_Q = copy.deepcopy(self.Q)

    @property
    def next_states_value(self):
        return self.target_Q(self._next_states).detach().max(1)[0].unsqueeze(1)

    def update_target_Q(self):
        if self.length[self._run].sum() % self.config.get('Q_update_interval') == 0:
            self.target_Q = copy.deepcopy(self.Q)
