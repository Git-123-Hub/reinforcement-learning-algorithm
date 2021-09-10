############################################
# @Author: Git-123-Hub
# @Date: 2021/9/9
# @Description: implementation of prioritized replay memory
############################################

import numpy as np

from utils.replayMemory import replayMemory
from utils.sumTree import sumTree
from utils.util import transfer_experience


class prioritizedMemory:
    """data structure that store experiences with priority and sample according to priority"""
    def __init__(self, capacity, batch_size, alpha, beta):
        self.capacity = capacity
        self.batch_size = batch_size
        # todo: check if the priority and memory use the same index
        self.priority = sumTree(capacity)  # store all the priority
        self.memory = replayMemory(capacity, batch_size)  # store all the experience

        # record the initial value of alpha and beta
        self._alpha = alpha
        self._beta = beta
        # todo: how to change its value
        self.alpha = alpha
        self.beta = beta

        # record the maximal priority for new experience to guarantee that all experience is seen at least once
        self.max_priority = 1

        # record all the index of experiences that have been sampled, make it easy for updating their priority
        self.sample_index = np.zeros(self.batch_size, dtype=int)

    def reset(self):
        self.priority.reset()
        self.memory.reset()
        self.alpha = self._alpha
        self.beta = self._beta
        self.max_priority = 1

    def add(self, experience, priority):
        self.memory.add(experience)
        self.max_priority = max(self.max_priority, priority)
        self.priority.add(self.max_priority ** self.alpha)

    def sample(self):
        experiences = np.zeros(self.batch_size, dtype=object)  # sampled experiences
        priorities = np.zeros(self.batch_size, dtype=np.float64)  # correspond priority
        # a) divide [0, `self.priority.sum`] into `batch_size` ranges
        segment = self.priority.sum / self.batch_size
        for i in range(self.batch_size):
            # print([segment * i, segment * (i + 1)])
            # b) uniformly sample a value from each range
            priority = np.random.uniform(segment * i, segment * (i + 1))
            # c) retrieve transition that corresponds to each value
            index, priority = self.priority.find(priority)
            # d) save sampled data
            self.sample_index[i] = index
            experiences[i] = self.memory[index]
            priorities[i] = priority
        sample_probabilities = priorities / self.priority.sum
        IS_weights = np.power(len(self.memory) * sample_probabilities, -self.beta)
        IS_weights /= IS_weights.max()
        return transfer_experience(experiences), IS_weights

    def update(self, td_errors):
        for i, index in enumerate(self.sample_index):
            # `i` represents the `i`th experience sampled, we use it to get the `i`th td_error for updating
            # `index` represents the index of the experience from `self.memory`
            # which is also the index of its priority in `self.priority`
            self.priority.update(index, abs(td_errors[i]) ** self.alpha)

    @property
    def ready(self):
        return self.memory.ready

    def __getitem__(self, index):
        if isinstance(index, slice):
            # todo: use slice to get memory and its priority
            start = index.start + self.capacity - 1
            index = slice(start, None, None)
            return self.memory[index]
        elif isinstance(index, int):
            experience = self.memory[index]
            priority = self.priority[index]
            return experience, priority
        else:
            raise NotImplementedError
