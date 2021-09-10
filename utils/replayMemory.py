############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: implementation of replay memory for DQN
############################################
import numpy as np

from utils.util import transfer_experience


class replayMemory:
    """data structure where we store the agent's experience, and sample them for the agent to learn"""

    def __init__(self, capacity, batch_size):
        assert capacity > batch_size, 'capacity should be greater than batch size'
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = np.zeros(capacity, dtype=object)  # len(self.memory) equals the capacity
        self._index = 0  # current position for adding new experience
        self._size = 0  # record the number of the all the experiences stored

    def reset(self):
        """clear all the experience that has been stored"""
        self.memory.fill(0)
        self._index = 0
        self._size = 0

    def add(self, experience):
        """
        add an experience to the memory
        :param experience: each experience is a tuple (state, action, reward, next_state, done)
        """
        self.memory[self._index] = np.array(experience, dtype=object)
        self._index = (self._index + 1) % len(self.memory)
        if self._size < len(self.memory):
            self._size += 1

    def sample(self):
        # sample all the current useful index without duplicate(replace=False)
        indices = np.random.choice(self._size, size=self.batch_size, replace=False)
        experiences = self.memory[indices]
        return transfer_experience(experiences)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        assert type(index) == int
        return self.memory[index]

    @property
    def ready(self):
        """return a bool indicate whether this ReplayMemory is ready to be sampled"""
        return self._size >= self.batch_size




