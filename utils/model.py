############################################
# @Author: Git-123-Hub
# @Date: 2021/10/17
# @Description: base class for all the network used in this project
############################################
import abc
from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


# NOTE that all the networks in this project are MLP(Multi-layer Perception)
class MLP(nn.Module, abc.ABC):
    """base class for all the network used in this project"""

    def __init__(self, input_size, output_size, hidden_layer: List[int] = None, *, activation=None, last_output=None):
        """
        construct a MLP given input_size, output_size, hidden-layer
        :type hidden_layer: specify size of each hidden layer
        :type activation: inner activation layer between hidden(linear) layer, default: nn.ReLU()
        :type last_output: layer before the last output, could be softmax or tanh
        """
        super(MLP, self).__init__()
        if hidden_layer is None:
            hidden_layer = []
        else:  # consider that we will change `hidden_layer`, it's better to separate it out
            hidden_layer = deepcopy(hidden_layer)

        if activation is None: activation = nn.ReLU()  # if not specified, use ReLu() as the inner activation

        # convert dimension from `input_size` to `hidden_layer` to `output_size`
        hidden_layer.insert(0, input_size)
        hidden_layer.append(output_size)
        modules = []
        for i in range(len(hidden_layer) - 1):
            modules.append(nn.Linear(hidden_layer[i], hidden_layer[i + 1]))
            if i != len(hidden_layer) - 2:  # the last layer don't need an activation function
                modules.append(activation)

        if last_output is not None:
            modules.append(last_output)

        self.net = nn.Sequential(*modules)

    @abc.abstractmethod
    def forward(self, *args):
        """different network act differently, this method should be implemented by subclass"""
        raise NotImplementedError

    def grad_info(self):
        for n, p in self.named_parameters():
            print(p.grad.min(), p.grad.max(), p.grad.mean(), p.grad.std())

    def init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=0.1)
                nn.init.constant_(layer.bias, 0.)


class QNet(MLP):
    """approximator for Q-value, output estimated value of each action"""

    def __init__(self, state_dim, action_dim, hidden_layer: List[int] = None, *, activation=None):
        """
        net that takes state as input and output Q-value of each action
        """
        super(QNet, self).__init__(state_dim, action_dim, hidden_layer, activation=activation)

    def forward(self, state):
        """input: state, output: Q-value of each action"""
        return self.net(state)


class StateCritic(MLP):
    """approximator of V-function, i.e. V(s)"""

    def __init__(self, state_dim, hidden_layer=None, *, activation=None):
        """net that takes state as input and output state-value of the input state"""
        super(StateCritic, self).__init__(state_dim, 1, hidden_layer, activation=activation)

    def forward(self, state):
        return self.net(state)


class StateActionCritic(MLP):
    """critic for state-action value, i.e. Q(s,a)"""

    def __init__(self, state_dim, action_dim, hidden_layer=None, *, activation=None):
        super(StateActionCritic, self).__init__(state_dim + action_dim, 1, hidden_layer, activation=activation)

    def forward(self, state, action):
        """return approximate value for Q(s,a)"""
        return self.net(torch.cat([state, action], dim=1))


class DeterministicActor(MLP):
    """actor with deterministic policy, i.e. action = policy(state)"""

    def __init__(self, state_dim, action_dim, hidden_layer=None, *, activation=None, max_action=1):
        """deterministic policy which takes state as input and output an action with value"""
        # NOTE that use Tanh() to scale the output to [-1, 1]
        super(DeterministicActor, self).__init__(state_dim, action_dim, hidden_layer,
                                                 activation=activation, last_output=nn.Tanh())
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action


class DiscreteStochasticActor(MLP):
    """actor with stochastic policy in a `discrete` scenario, where action is bounded to a range of specified value"""

    def __init__(self, state_dim, action_dim, hidden_layer=None, *, activation=None):
        """stochastic policy which takes state as input and output probability of choosing each action"""
        # NOTE that use Softmax to get normalized probability
        super(DiscreteStochasticActor, self).__init__(state_dim, action_dim, hidden_layer,
                                                      activation=activation, last_output=nn.Softmax(dim=1))

    def forward(self, state):
        """The action returned is sampled according to corresponding distribution"""
        probs = self.net(state)  # probability of each action
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class ContinuousStochasticActor(MLP):
    """stochastic actor in a `continuous` scenario, where action is sampled from Gaussian distribution constructed"""

    def __init__(self, state_dim, action_dim, hidden_layer=None, *, activation=None, max_action=1,
                 fix_std: float = None):
        """stochastic actor that take state as input and output mean and log_std of the action distribution"""
        assert hidden_layer is not None, 'hidden layer must be provided for ContinuousStochasticActor'
        super(ContinuousStochasticActor, self).__init__(state_dim, hidden_layer[-1], hidden_layer[:-1],
                                                        activation=activation)

        self.mean_output = nn.Linear(hidden_layer[-1], action_dim)
        if fix_std:
            self.log_std_output = None
            self.std = torch.tensor(fix_std * np.ones(action_dim, dtype=np.float32))
        else:
            self.log_std_output = nn.Linear(hidden_layer[-1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        """The action is sampled from the Gaussian distribution using mean and log_std from the network"""
        # NOTE that the output is a distribution
        base = self.net(state)
        mean = self.mean_output(base).squeeze(0)
        if self.log_std_output is None:
            std = self.std
        else:
            log_std = self.log_std_output(base).squeeze(0)
            log_std.clip_(-20, 2)
            # NOTE that the clip value of log_std is taken from rlKit
            # (https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/policies/gaussian_policy.py)
            std = torch.exp(log_std)
        return Normal(mean * self.max_action, std)

    def init(self):
        super(ContinuousStochasticActor, self).init()

        nn.init.normal_(self.mean_output.weight, mean=0., std=0.1)
        nn.init.constant_(self.mean_output.bias, 0.)

        if self.log_std_output is not None:
            nn.init.normal_(self.log_std_output.weight, mean=0., std=0.1)
            nn.init.constant_(self.log_std_output.bias, 0.)


if __name__ == '__main__':
    # some test case to see the structure of the network
    s_dim, a_dim, h_layer = 12, 2, [64, 32]
    print('QNet:', QNet(s_dim, a_dim, h_layer).net)
    print('StateCritic:', StateCritic(s_dim, h_layer).net)
    print('StateActionCritic:', StateActionCritic(s_dim, a_dim, h_layer).net)
    print('DeterministicActor:', DeterministicActor(s_dim, a_dim, h_layer).net)
    print('DiscreteStochasticActor:', DiscreteStochasticActor(s_dim, a_dim, h_layer).net)
    actor = ContinuousStochasticActor(s_dim, a_dim, h_layer)
    actor.init()
    print('ContinuousStochasticActor:', actor.net)
