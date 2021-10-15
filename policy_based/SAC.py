############################################
# @Author: Git-123-Hub
# @Date: 2021/10/12
# @Description: implementation of SAC(soft actor critic)
############################################
import os
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn.functional as F

from utils import Agent, replayMemory
from utils.util import soft_update


# NOTE that this file is based on the updated version of SAC
# Soft Actor-Critic Algorithms and Applications (https://arxiv.org/abs/1812.05905)


class SAC(Agent):
    """off-policy maximum entropy deep reinforcement learning with a stochastic actor"""

    def __init__(self, env, actor, critic, config):
        super(SAC, self).__init__(env, config)
        self._actor = actor
        self._critic = critic

        # these are initialized in `run_reset()`
        self.actor, self.actor_optimizer = None, None
        self.critic1, self.target_critic1, self.critic1_optimizer = None, None, None
        self.critic2, self.target_critic2, self.critic2_optimizer = None, None, None

        # entropy and alpha setting
        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        # NOTE that this is the heuristic value of target entropy from Appendix D of the paper

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.get('learning_rate', 1e-4))
        self.alpha = self.log_alpha.exp()
        # NOTE that we are optimizing log value of alpha and get alpha through exp()
        # it's a bit different from the paper but I found this way in many implementation, including the official one.
        # I also found the discussion here:
        # https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability
        # which probably be the reason, but I'm not quite sure

        self.replayMemory = replayMemory(self.config.get('memory_capacity', 20000), self.config.get('batch_size', 256))

        self.max_action = self.env.action_space.high[0]
        self.min_action = self.env.action_space.low[0]

    def run_reset(self):
        super(SAC, self).run_reset()

        self.actor = self._actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 1e-4))

        self.critic1 = self._critic(self.state_dim, self.action_dim)
        self.target_critic1 = deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.get('learning_rate', 1e-4))

        self.critic2 = self._critic(self.state_dim, self.action_dim)
        self.target_critic2 = deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.get('learning_rate', 1e-4))

    def select_action(self):
        """sample action from the policy"""
        # todo: check data dimension
        self.actor.eval()
        # todo: should select action using `tanh`
        self.action = self.actor(self.state).sample().detach().numpy()
        self.actor.train()

    def learn(self):
        if not self.replayMemory.ready:  # only start to learn when there are enough experience to sample
            return
        _total_t = time()
        states, actions, rewards, next_states, dones = self.replayMemory.sample()

        self.logger.info('-' * 20)

        # a) update Critic by minimizing the soft Bellman residual
        _critic_time = time()

        with torch.no_grad():
            next_actions, log_prob = self.resample(next_states)
            # calculate the soft-Q-value of (next-states, next-actions) using corresponding target network
            soft_Q_value_1 = self.target_critic1(next_states, next_actions)
            soft_Q_value_2 = self.target_critic2(next_states, next_actions)
            soft_state_value = torch.min(soft_Q_value_1, soft_Q_value_2) - self.alpha * log_prob  # V(s_t+1)
            # calculate target value for critic
            target_value = rewards + (1 - dones) * self.config.get('discount_factor', 0.99) * soft_state_value

        # update critic_1
        current_soft_Q_value_1 = self.critic1(states, actions)
        loss = F.mse_loss(current_soft_Q_value_1, target_value)
        self.critic1_optimizer.zero_grad()
        loss.backward()
        self.critic1_optimizer.step()

        # update critic_2
        current_soft_Q_value_2 = self.critic2(states, actions)
        loss = F.mse_loss(current_soft_Q_value_2, target_value)
        self.critic2_optimizer.zero_grad()
        loss.backward()
        self.critic2_optimizer.step()

        self.logger.info(f'critic time: {time() - _critic_time}')

        # b) update policy
        _policy_time = time()
        actions, log_prob = self.resample(states)
        soft_Q_value_1 = self.critic1(states, actions)
        soft_Q_value_2 = self.critic2(states, actions)
        soft_Q_value = torch.min(soft_Q_value_1, soft_Q_value_2)
        # todo: necessity of mean
        loss = (self.alpha * log_prob - soft_Q_value).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.logger.info(f'policy time: {time() - _policy_time}')

        # c) update alpha
        _alpha_time = time()
        loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        # NOTE that we are optimizing log_alpha, so the alpha in eq(18) should be replaced with log_alpha(I guess)
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # self.logger.info(f'alpha time: {time() - _alpha_time}')

        # d) update target network
        _soft_time = time()
        # todo: maybe add interval
        soft_update(self.critic1, self.target_critic1, self.config.get('tau', 0.01))
        soft_update(self.critic2, self.target_critic2, self.config.get('tau', 0.01))

        # self.logger.info(f'soft update time: {time() - _soft_time}')

        self.logger.info(f'learn time: {time() - _total_t}')

    def resample(self, state):
        """sample action on state from the policy, and calculate the log-prob using reparameterization trick"""
        _t = time()
        # calculate the soft-state-value of `state`
        distribution = self.actor(state)
        # get next_action of next_states using the reparameterization trick
        # todo: maybe update critic doesn't need reparameterization, just the policy
        # todo: check data dimension
        mu = distribution.rsample()
        action = torch.tanh(mu)
        # calculate log_prob of next_actions using the method introduced in Appendix C of the paper
        # todo: use mean or sum
        log_prob = distribution.log_prob(mu)
        # todo: how to calculate `pow` and `sum`
        log_prob -= torch.log(1 - action.pow(2) + torch.finfo(torch.float32).eps)

        self.logger.info(f'resample time: {time() - _t}')
        return action, log_prob.sum(dim=1, keepdim=True)

    def save_policy(self):
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.actor is None: self.actor = self._actor(self.state_dim, self.action_dim)
        self.actor.load_state_dict(torch.load(file))
        self.actor.eval()

    def test_action(self, state):
        # NOTE that deterministic action use mean of the distribution
        # todo: might need `tanh`
        self.action = self.actor(self.state).mean.detach().numpy()
