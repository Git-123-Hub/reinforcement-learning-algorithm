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
        self.replay_buffer = prioritizedMemory(self.config.memory_capacity, self.config.batch_size,
                                               self.config.alpha, self.config.beta)

    def episode_reset(self):
        """implement parameter(alpha, beta) decay before episode starts"""
        super(DDQN_PER, self).episode_reset()
        # todo: linear change of parameter alpha, beta of the prioritized replay memory

    def learn(self):
        if len(self.replay_buffer) < self.config.random_steps:
            # interact with the env randomly to generate experience before start to learn
            # only start to learn when there are enough experiences to sample
            return

        (states, actions, rewards, next_states, dones), IS_weights = self.replay_buffer.sample()

        current_state_value = self.Q(states).gather(1, actions.long()).squeeze(1).to(self.device)  # shape: batch_size
        next_state_value = self.get_next_state_value(next_states)  # shape: batch_size
        assert rewards.shape == next_state_value.shape == dones.shape == current_state_value.shape
        target_value = rewards + self.config.gamma * next_state_value * (1 - dones)  # shape: batch_size

        # update priority with td-error
        td_errors = target_value - current_state_value  # shape: batch_size
        self.replay_buffer.update(td_errors.tolist())

        # calculate loss using IS_weights
        loss = F.mse_loss(current_state_value, target_value, reduction='none').squeeze()
        IS_weights = torch.from_numpy(IS_weights)
        loss = (loss * IS_weights).mean()  # element-wise multiplication
        self.logger.info(f'loss: {loss.item()}')

        self.gradient_descent(loss)
        self.update_target_network()
