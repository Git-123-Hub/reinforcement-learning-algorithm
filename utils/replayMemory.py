############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: implementation of replay memory for DQN
############################################
import numpy as np
import torch


class replayMemory:
    """data structure where we store the agent's experience, and sample from them for the agent to learn"""

    def __init__(self, capacity, batch_size):
        assert int(capacity) > batch_size, 'capacity should be greater than batch size'
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)

        self.state = np.zeros(int(capacity), dtype=object)
        self.action = np.zeros(int(capacity), dtype=object)
        self.reward = np.zeros(int(capacity), dtype=float)
        self.next_state = np.zeros(int(capacity), dtype=object)
        self.done = np.zeros(int(capacity), dtype=bool)  # type for done

        self._index = 0  # current position for adding new experience
        self._size = 0  # record the number of the all the experiences stored
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """clear all the experience that has been stored, reset the replay memory"""
        self.state.fill(0)
        self.action.fill(0)
        self.reward.fill(0)
        self.next_state.fill(0)
        self.done.fill(0)

        self._index = 0
        self._size = 0

    def add(self, state, action, reward, next_state, done):
        """ add an experience to the memory """
        self.state[self._index] = state
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_state[self._index] = next_state
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, size=None):
        if size is None: size = self.batch_size
        # sample all the current useful index without duplicate(replace=False)
        indices = np.random.choice(self._size, size=size, replace=False)
        return self.transfer_experience(indices)

    def transfer_experience(self, indices):
        """transfer the data of `indices` from ndarray to tensor and change the shape of the data if necessary"""
        # retrieve data using `__getitem__`, Note that the data stored is ndarray
        states, actions, rewards, next_states, dones = self[indices]
        # NOTE that `states`, `actions`, `next_states` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.from_numpy(np.vstack(actions)).float().to(self.device)  # torch.Size([batch_size, action_dim])
        rewards = torch.from_numpy(rewards).float().to(self.device)  # just a tensor with length: batch_size
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)  # Size([batch_size, state_dim])
        dones = torch.from_numpy(dones).float().to(self.device)  # just a tensor with length: batch_size
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        state = self.state[index]
        action = self.action[index]
        reward = self.reward[index]
        next_state = self.next_state[index]
        done = self.done[index]
        return state, action, reward, next_state, done
