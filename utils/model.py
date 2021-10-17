############################################
# @Author: Git-123-Hub
# @Date: 2021/10/17
# @Description: base class for all the network used in this project
############################################
import abc
from typing import List

import torch
from torch import nn
from torch.distributions import Categorical, Normal


# NOTE that all the networks in this project are MLP(Multilayer Perception)
# and I use RELU as activation function for all of them
class MLP(nn.Module, abc.ABC):
    """base class for all the network used in this project"""

    def __init__(self, input_size, output_size, hidden_layer: List[int] = None, *, softmax=False):
        """
        construct a MLP given input_size, output_size, hidden-layer
        :type softmax: bool, indicate whether add softmax to the last output of the net
        """
        super(MLP, self).__init__()
        if hidden_layer is None:
            hidden_layer = [32, 32]
        modules = []
        hidden_layer.insert(0, input_size)
        hidden_layer.append(output_size)
        # convert dimension from `input_size` to `hidden_layer` to `output_size`
        for i in range(len(hidden_layer) - 1):
            modules.append(nn.Linear(hidden_layer[i], hidden_layer[i + 1]))
            if i != len(hidden_layer) - 2:  # the last layer don't need an activation function
                modules.append(nn.ReLU())

        if softmax:
            modules.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*modules)

    @abc.abstractmethod
    def forward(self, *args):
        """different network act differently, this method should be implemented by subclass"""
        raise NotImplementedError


class StateActionCritic(MLP):
    """approximator for Q-value, i.e. state-action value, Q(s,a)"""

    def __init__(self, state_dim, action_dim, *, hidden_layer: List[int] = None):
        """
        net that takes state as input and output Q-value of each action
        :type hidden_layer: specify size of each hidden layer
        """
        super(StateActionCritic, self).__init__(state_dim, action_dim, hidden_layer)

    def forward(self, state):
        """input: state, output: Q-value of each action"""
        return self.net(state)


class StateCritic(MLP):
    """approximator of V-function, i.e. V(s)"""

    def __init__(self, state_dim, *, hidden_layer=None):
        """net that takes state as input and output state-value of input state"""
        super(StateCritic, self).__init__(state_dim, 1, hidden_layer)

    def forward(self, state):
        return self.net(state)


class DeterministicActor(MLP):
    """actor with deterministic policy, i.e. action = policy(state)"""

    def __init__(self, state_dim, action_dim, *, hidden_layer=None):
        """deterministic policy which takes state as input and output an action with value"""
        super(DeterministicActor, self).__init__(state_dim, action_dim, hidden_layer)

    def forward(self, state):
        return self.net(state)


class DiscreteStochasticActor(MLP):
    """actor with stochastic policy in a `discrete` scenario where action is bounded to a range of specified value"""

    def __init__(self, state_dim, action_dim, *, hidden_layer=None):
        """stochastic policy which takes state as input and output probability of choosing each action"""
        super(DiscreteStochasticActor, self).__init__(state_dim, action_dim, hidden_layer, softmax=True)

    def forward(self, state):
        """The action returned is sampled according to corresponding probability"""
        probs = self.net(state)  # probability of each action
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class ContinuousStochasticActor(MLP):
    """stochastic actor in a `continuous` scenario where action is sampled from Gaussian distribution constructed"""

    def __init__(self, state_dim, action_dim, *, hidden_layer=None):
        """stochastic actor that take state as input and output mean and log_std of the action distribution"""
        super(ContinuousStochasticActor, self).__init__(state_dim, hidden_layer[-1], hidden_layer[:-1])

        self.mean_output = nn.Linear(hidden_layer[-1], action_dim)
        self.log_std_output = nn.Linear(hidden_layer[-1], action_dim)

    def forward(self, state):
        """The action is sampled from the Gaussian distribution using mean and log_std from the network"""
        base = self.net(state)
        mean, log_std = self.mean_output(base).squeeze(0), self.log_std_output(base).squeeze(0)
        log_std.clip_(-20, 2)
        # NOTE that the clip value of log_std is taken from rlKit
        # (https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/policies/gaussian_policy.py)
        std = torch.exp(log_std)
        # NOTE that the output is a distribution
        return Normal(mean, std)


if __name__ == '__main__':
    print('net structure of StateActionCritic:', StateActionCritic(12, 2, hidden_layer=[64, 32]).net)
    print('net structure of StateCritic:', StateCritic(12, hidden_layer=[64, 32]).net)
    print('net structure of DeterministicActor:', DeterministicActor(12, 2, hidden_layer=[64, 32]).net)
    print('net structure of DiscreteStochasticActor:', DiscreteStochasticActor(12, 2, hidden_layer=[64, 32]).net)
    print('net structure of ContinuousStochasticActor:', ContinuousStochasticActor(12, 2, hidden_layer=[64, 32]).net)
