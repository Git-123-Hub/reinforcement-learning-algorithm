############################################
# @Author: Git-123-Hub
# @Date: 2021/9/9
# @Description: implementation of prioritized replay memory
############################################

import numpy as np

from utils.replayMemory import replayMemory
from utils.sumTree import sumTree
from utils.util import transfer_experience


class prioritizedMemory(replayMemory):
    """data structure that store experiences with priority and sample according to priority"""

    def __init__(self, capacity, batch_size, alpha, beta):
        super(prioritizedMemory, self).__init__(capacity, batch_size)
        self.priority = sumTree(capacity)  # store all the priority

        # record the initial value of alpha and beta
        self.alpha0 = alpha
        self.beta0 = beta
        self.alpha = alpha
        self.beta = beta

        # record the maximal priority for new experience to guarantee that all experience is seen at least once
        self.max_priority = 1

        # record all the index of experiences that have been sampled, make it easy for updating their priorities
        self.sample_index = np.zeros(self.batch_size, dtype=int)

    def reset(self):
        super(prioritizedMemory, self).reset()
        self.priority.reset()
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.max_priority = 1

    def add(self, state, action, reward, next_state, done):
        """add experience and use `self.max_priority` to initialize it's priority"""
        super(prioritizedMemory, self).add(state, action, reward, next_state, done)
        self.priority.add(self.max_priority)

    def sample(self, size=None):
        if size is None: size = self.batch_size
        sample_priorities = np.zeros(size, dtype=np.float64)  # correspond priority
        # a) divide [0, `self.priority.sum`] into `batch_size` ranges
        segment = self.priority.sum / size
        for i in range(size):
            # b) uniformly sample a value from each range
            priority = np.random.uniform(segment * i, segment * (i + 1))
            # c) retrieve transition that corresponds to each value
            index, priority = self.priority.find(priority)
            # d) save sampled index
            self.sample_index[i] = index
            sample_priorities[i] = priority
        sample_probabilities = sample_priorities / self.priority.sum
        IS_weights = np.power(self.capacity * sample_probabilities, -self.beta)
        IS_weights /= IS_weights.max()
        return self.transfer_experience(self.sample_index), IS_weights

    def update(self, td_errors):
        for i, index in enumerate(self.sample_index):
            # `i` represents the `i`th experience sampled, we use it to get the `i`th td_error for updating
            # `index` represents the index of the experience from `self.memory`
            # which is also the index of its priority in `self.priority`
            priority = abs(td_errors[i]) ** self.alpha
            self.priority[index] = priority
            self.max_priority = max(self.max_priority, priority)

