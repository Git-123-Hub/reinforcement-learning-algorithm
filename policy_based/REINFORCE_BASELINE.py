############################################
# @Author: Git-123-Hub
# @Date: 2021/9/22
# @Description: implementation of REINFORCE_with_BASELINE
############################################

import os

import torch
import torch.nn.functional as F
from torch import optim

from utils.Agent import Agent
from utils.util import discount_sum


# cite from "Reinforcement Learning An Introduction" Sec 13.5:
# Although the REINFORCE-with-baseline method learns both a policy and a state-value function,
# we do not consider it to be an actor–critic method
# because its state-value function is used only as a baseline, not as a critic.
# That is, it is not used for bootstrapping
# (updating the value estimate for a state from the estimated values of subsequent states),
# but only as a baseline for the state whose estimate is being updated.
# This is a useful distinction, for only through bootstrapping do we introduce bias and
# an asymptotic dependence on the quality of the function approximation.


class REINFORCE_BASELINE(Agent):
    def __init__(self, env, actor, critic, config):
        super(REINFORCE_BASELINE, self).__init__(env, config)
        self._actor = actor
        self.actor, self.actor_optimizer = None, None

        self._critic = critic
        self.critic, self.critic_optimizer = None, None

        self.episode_reward = []  # reward of each step in a whole episode, used to calculate sum of discounted reward
        self.episode_log_prob = []  # log probability of each action in a whole episode, used to update actor
        self.episode_state_value = []  # value of each state in a whole episode, used as baseline

    def run_reset(self):
        super(REINFORCE_BASELINE, self).run_reset()
        self.actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('learning_rate', 0.001))

        self.critic = self._critic(self.state_dim, self.config.get('critic_hidden_layer'))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get('learning_rate', 0.001))

    def episode_reset(self):
        super(REINFORCE_BASELINE, self).episode_reset()
        self.episode_reward, self.episode_log_prob, self.episode_state_value = [], [], []

    def select_action(self):
        state = torch.tensor(self.state).float().unsqueeze(0)
        self.action, log_prob = self.actor(state)
        self.episode_log_prob.append(log_prob)

    def learn(self):
        # NOTE that this algorithm only learn when an episode finishes
        # before that, we need to collect sequence of reward and state-value of this episode
        self.episode_reward.append(self.reward)
        state = torch.tensor(self.state).float().unsqueeze(0)
        self.episode_state_value.append(self.critic(state))

        if not self.done:  # only learn when an episode finishes
            return

        returns = discount_sum(self.episode_reward, self.config.get('discount_factor', 0.99), normalize=True)

        # update actor
        advantage_list = returns - torch.cat(self.episode_state_value).squeeze()
        policy_loss_list = torch.cat(self.episode_log_prob) * -advantage_list.detach()  # note the negative sign
        loss = policy_loss_list.sum()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # update critic
        loss = F.mse_loss(torch.cat(self.episode_state_value).squeeze(), returns, reduction='sum').float()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def save_policy(self):
        """save the parameter of the policy network(`self.Q) when the running reward reaches `self.goal`"""
        if self._running_reward >= self.goal:
            name = f'{self.__class__.__name__}_solve_{self.env_id}_{self._run + 1}_{self._episode + 1}.pt'
            torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

    def load_policy(self, file):
        if self.actor is None:
            self.actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'))
        self.actor.load_state_dict(torch.load(file))
        self.actor.eval()

    def test_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad(): action, _ = self.actor(state)
        return action
