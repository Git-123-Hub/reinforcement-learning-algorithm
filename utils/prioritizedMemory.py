############################################
# @Author: Git-123-Hub
# @Date: 2021/9/9
# @Description: implementation of prioritized replay memory
############################################
from random import random

import numpy as np
import torch

from utils.replayMemory import replayMemory
from utils.sumTree import sumTree


class prioritizedMemory:
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
        experiences = np.zeros(self.batch_size, dtype=object)
        priorities = np.zeros(self.batch_size, dtype=np.float64)
        segment = self.priority.sum / self.batch_size
        for i in range(self.batch_size):
            # print([segment * i, segment * (i + 1)])
            priority = np.random.uniform(segment * i, segment * (i + 1))
            index, priority = self.priority.find(priority)
            self.sample_index[i] = index
            experiences[i] = self.memory[index]
            priorities[i] = priority
        # todo: transfer experience

        sample_probabilities = priorities / self.priority.sum
        IS_weights = np.power(len(self.memory) * sample_probabilities, -self.beta)
        IS_weights /= IS_weights.max()
        return transfer_experience(experiences), IS_weights

    def update(self, td_errors):
        td_errors = td_errors.squeeze().tolist()
        for i, index in enumerate(self.sample_index):
            # {i} represents the {i}th experience sampled, we use it to get the {i}th td_error for updating
            # {index} represents the index of the experience from `self.memory`
            # which is also the index of priority in `self.priority`
            self.priority.update(index, abs(td_errors[i]) ** self.alpha)

    @property
    def ready(self):
        return self.memory.ready


# todo: change to util
def transfer_experience(experiences):
    experiences = np.vstack(experiences).transpose()
    # so we first stack the sampled experiences vertically,
    # so that each row represents a tuple for experience, and same component(s,a,r,d,s') stay in the same index
    # after transpose, states are transferred to the first row, actions on the second row...

    # transfer array to tensor to pass the the network
    # Note that the first dimension has to be batch size, so i use np.vstack before convert to tensor
    states = torch.from_numpy(np.vstack(experiences[0])).float()
    actions = torch.from_numpy(np.vstack(experiences[1])).float()
    rewards = torch.from_numpy(np.vstack(experiences[2])).float()
    next_states = torch.from_numpy(np.vstack(experiences[3])).float()
    dones = torch.from_numpy(np.vstack(experiences[4])).float()
    return states, actions, rewards, next_states, dones
