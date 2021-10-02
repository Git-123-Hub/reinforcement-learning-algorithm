############################################
# @Author: Git-123-Hub
# @Date: 2021/9/8
# @Description: implementation of Double Deep Q-Learning with prioritized experience replay
############################################
from utils import prioritizedMemory
from value_based import DDQN
import torch.nn.functional as F
import torch


class DDQN_PER(DDQN):
    def __init__(self, env, Q_net, config):
        super(DDQN_PER, self).__init__(env, Q_net, config)
        # todo: how to write configure
        self.replayMemory = prioritizedMemory(
            self.config['memory_capacity'],
            self.config['batch_size'], self.config['alpha'], self.config['beta'])

    def episode_reset(self):
        """implement parameter(alpha, beta) decay before episode starts"""
        super(DDQN_PER, self).episode_reset()
        # linear change of parameter alpha, beta of the prioritized replay memory
        self.replayMemory.alpha = self.replayMemory.alpha0 + (1 - self.replayMemory.alpha0) * self._episode / (
                self.episode_num - 1)
        self.replayMemory.beta = self.replayMemory.beta0 + (1 - self.replayMemory.beta0) * self._episode / (
                self.episode_num - 1)
        # when `self.episode` equals 0(i.e. each episode starts),
        # alpha and beta are set to its initial value, which is provided in the config
        # when `self.episode` equals `self.episode_num`-1(i.e. each episode stops), alpha and beta are set to 1

    def learn(self):
        experiences, IS_weights = self.replayMemory.sample()
        states, actions, rewards, next_states, dones = experiences

        current_state_value = self.Q(states).gather(1, actions.long())
        next_state_value = self.get_next_state_value(next_states)
        target_value = rewards + self.config.get('discount_factor', 0.99) * next_state_value * (1 - dones)

        # update priority with td-error
        td_errors = target_value - current_state_value
        self.replayMemory.update(td_errors.squeeze().tolist())

        # calculate loss using IS_weights
        loss = F.mse_loss(current_state_value, target_value, reduction='none').squeeze()
        IS_weights = torch.from_numpy(IS_weights)
        loss = (loss * IS_weights).mean()  # element-wise multiplication
        self.logger.info(f'loss: {loss.item()}')

        self.gradient_descent(loss)
        self.update_target_network()
