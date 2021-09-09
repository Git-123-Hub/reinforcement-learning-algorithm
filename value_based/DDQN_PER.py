############################################
# @Author: Git-123-Hub
# @Date: 2021/9/8
# @Description: implementation of Double Deep Q-Learning with prioritized experience replay
############################################
from utils.prioritizedMemory import prioritizedMemory
from value_based.DDQN import DDQN
import torch.nn.functional as F


class DDQN_PER(DDQN):
    def __init__(self, env, Q_net, config):
        super(DDQN_PER, self).__init__(env, Q_net, config)
        # todo: how to write configure
        self.replayMemory = prioritizedMemory(self.config['memory_capacity'],
                                              self.config['batch_size'], self.config['alpha'], self.config['beta'])

    def save_experience(self):
        experience = (self.state, self.action, self.reward, self.next_state, self.done)
        priority = self.replayMemory.max_priority
        self.replayMemory.add(experience, priority)

    def learn(self):
        experiences, IS_weights = self.replayMemory.sample()
        self._states, self._actions, self._rewards, self._next_states, self._dones = experiences
        td_errors = self.target_value - self.current_states_value
        self.replayMemory.update(td_errors.squeeze().tolist())
        # calculate loss using IS_weights
        loss = F.mse_loss(self.current_states_value, self.target_value)
        self.perform_gradient_descent(loss)
        self.update_target_Q()
