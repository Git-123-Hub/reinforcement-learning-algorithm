############################################
# @Author: Git-123-Hub
# @Date: 2021/10/17
# @Description: base class for all the network used in this project
############################################
import abc
from copy import deepcopy
from typing import List

import torch
from torch import nn
from torch.distributions import Categorical, Normal


# NOTE that all the networks in this project are MLP(Multilayer Perception)
# and I use RELU as activation function for all of them
class MLP(nn.Module, abc.ABC):
    """base class for all the network used in this project"""

    def __init__(self, input_size, output_size, hidden_layer: List[int] = None, *, softmax=False, tanh=False):
        """
        construct a MLP given input_size, output_size, hidden-layer
        :type softmax: bool, indicate whether add softmax to the last output of the net
        :type tanh: bool, indicate whether add tanh to bound the output to [-1, 1]
        """
        super(MLP, self).__init__()
        if hidden_layer is None:
            hidden_layer = [32, 32]
        else:  # consider that we will change `hidden_layer`, it's better to separate it out
            hidden_layer = deepcopy(hidden_layer)
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
        if tanh:
            modules.append(nn.Tanh())

        self.net = nn.Sequential(*modules)

    @abc.abstractmethod
    def forward(self, *args):
        """different network act differently, this method should be implemented by subclass"""
        raise NotImplementedError


class QNet(MLP):
    """approximator for Q-value, output estimated value of each action"""

    def __init__(self, state_dim, action_dim, hidden_layer: List[int] = None):
        """
        net that takes state as input and output Q-value of each action
        :type hidden_layer: specify size of each hidden layer
        """
        super(QNet, self).__init__(state_dim, action_dim, hidden_layer)

    def forward(self, state):
        """input: state, output: Q-value of each action"""
        return self.net(state)


class StateCritic(MLP):
    """approximator of V-function, i.e. V(s)"""

    def __init__(self, state_dim, hidden_layer=None):
        """net that takes state as input and output state-value of input state"""
        super(StateCritic, self).__init__(state_dim, 1, hidden_layer)

    def forward(self, state):
        return self.net(state)


class StateActionCritic(MLP):
    """critic for state-action value, i.e. Q(s,a)"""

    def __init__(self, state_dim, action_dim, hidden_layer=None):
        super(StateActionCritic, self).__init__(state_dim + action_dim, 1, hidden_layer)

    def forward(self, state, action):
        """return approximate value for Q(s,a)"""
        # input shape of the net: [batch_size, state_dim + action_dim], output shape: [batch_size, 1]
        # so squeeze(dim=1) is needed to make sure each of the (state, action) pair corresponds to one critic value
        return self.net(torch.cat([state, action], dim=1)).squeeze(dim=1)


class DeterministicActor(MLP):
    """actor with deterministic policy, i.e. action = policy(state)"""

    def __init__(self, state_dim, action_dim, hidden_layer=None, *, max_action=1):
        """deterministic policy which takes state as input and output an action with value"""
        super(DeterministicActor, self).__init__(state_dim, action_dim, hidden_layer, tanh=True)
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action


class DiscreteStochasticActor(MLP):
    """actor with stochastic policy in a `discrete` scenario where action is bounded to a range of specified value"""

    def __init__(self, state_dim, action_dim, hidden_layer=None):
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

    def __init__(self, state_dim, action_dim, hidden_layer=None):
        """stochastic actor that take state as input and output mean and log_std of the action distribution"""
        super(ContinuousStochasticActor, self).__init__(state_dim, hidden_layer[-1], hidden_layer[:-1])

        self.mean_output = nn.Linear(hidden_layer[-1], action_dim)
        self.log_std_output = nn.Linear(hidden_layer[-1], action_dim)

    def forward(self, state):
        """The action is sampled from the Gaussian distribution using mean and log_std from the network"""
        # NOTE that `forward()` returns a distribution, so there is no need to multiply by max_action here
        base = self.net(state)
        mean, log_std = self.mean_output(base).squeeze(0), self.log_std_output(base).squeeze(0)
        log_std.clip_(-20, 2)
        # NOTE that the clip value of log_std is taken from rlKit
        # (https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/policies/gaussian_policy.py)
        std = torch.exp(log_std)
        # NOTE that the output is a distribution
        return Normal(mean, std)


if __name__ == '__main__':
    # some test case to see the structure of the network
    s_dim, a_dim, h_layer = 12, 2, [64, 32]
    print('QNet:', QNet(s_dim, a_dim, h_layer).net)
    print('StateCritic:', StateCritic(s_dim, h_layer).net)
    print('StateActionCritic:', StateActionCritic(s_dim, a_dim, h_layer).net)
    print('DeterministicActor:', DeterministicActor(s_dim, a_dim, h_layer).net)
    print('DiscreteStochasticActor:', DiscreteStochasticActor(s_dim, a_dim, h_layer).net)
    print('ContinuousStochasticActor:', ContinuousStochasticActor(s_dim, a_dim, h_layer).net)
