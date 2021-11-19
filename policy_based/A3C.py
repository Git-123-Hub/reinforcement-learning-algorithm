############################################
# @Author: Git-123-Hub
# @Date: 2021/11/10
# @Description: implementation of A3C(Asynchronous Advantage Actor Critic)
############################################
import datetime
import os
import random
import time
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils import Agent, EpisodicReplayMemory
import torch.multiprocessing as mp
from utils.const import Color, DefaultGoal
from utils.shared_adam import SharedAdam
from utils.util import setup_logger, discount_sum, initial_folder


class A3CWorker(mp.Process):
    """A3C worker that interact with environment asynchronously for an episode"""

    def __init__(self, env_id, global_actor, global_critic, actor_optimizer, critic_optimizer,
                 global_episode_num, total_episode_num, length,
                 episode_rewards, running_rewards, window,
                 n, gamma, render, run_num, policy_path):
        super(A3CWorker, self).__init__()
        # setup env
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.state, self.action, self.reward, self.next_state, self.done = None, None, None, None, False
        self.state_value, self.log_prob = None, None
        # these two value should also be stored in replay memory for learning
        self.episode_reward = 0
        # reward of the current episode, this value will be pushed to `self.global_episode_rewards`
        # when episode terminates, and will finally be used to plot reward with respect to episode num

        self.name = f'{self.__class__.__name__}-{n}'

        # setup network
        self.global_actor = global_actor
        self.global_critic = global_critic

        self.global_actor_optimizer = actor_optimizer
        self.global_critic_optimizer = critic_optimizer

        self.actor, self.critic = None, None
        self.actor_optimizer, self.critic_optimizer = None, None

        # self.actor_optimizer = actor_optimizer
        # self.critic_optimizer = critic_optimizer
        # self.actor_optimizer = Adam(global_actor.parameters(), lr=1e-3)
        # self.critic_optimizer = Adam(global_critic.parameters(), lr=1e-3)
        # NOTE that these optimizers are for global net
        # local net doesn't need to be optimized, it just copy from global

        self.global_episode_num = global_episode_num
        self.total_episode_num = total_episode_num
        self.current_episode_num = 0

        self.length = length

        self.episode_rewards = episode_rewards
        self.running_rewards = running_rewards
        self.window = window

        self.replayMemory = EpisodicReplayMemory(gamma)
        self.render = render
        self.policy_path = policy_path
        self.run_num = run_num

    def episode_reset(self):
        """operation before an episode starts"""
        self.current_episode_num = deepcopy(self.global_episode_num.value)
        # we also maintain `current_episode_num` so that when an episode stops, we can get `current` episode accurately
        # not the global value of that moment, which might have been changed by other workers

        self.replayMemory.reset()
        # copy from the global network
        self.actor = deepcopy(self.global_actor)
        self.critic = deepcopy(self.global_critic)

        # todo: should each process has its own optimizer
        # self.actor_optimizer = Adam(self.global_actor.parameters(), lr=1e-3)
        # self.critic_optimizer = Adam(self.global_critic.parameters(), lr=1e-3)

        self.actor_optimizer = self.global_actor_optimizer
        self.critic_optimizer = self.global_critic_optimizer

        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0  # reset reward of the current episode

    def select_action(self):
        # with torch.no_grad():
        state = torch.tensor(self.state).float().unsqueeze(0)
        action_dist = self.actor(state)
        self.action = action_dist.sample().detach()
        self.log_prob = action_dist.log_prob(self.action).sum()
        # todo: max action should be specified in the network
        self.action = self.action.numpy()
        self.state_value = self.critic(state).squeeze()  # shape: torch.Size([]), just a tensor

    def run(self):
        """
        this is the function that will be executed when `worker.start()` is invoked.
        in our situation, it should be the training method
        """
        while self.global_episode_num.value < self.total_episode_num:
            self.global_episode_num.value += 1
            self.episode_reset()
            length = 0

            # agent interact with env for an episode
            while not self.done:
                length += 1
                self.select_action()
                # execute action
                self.next_state, self.reward, self.done, _ = self.env.step(self.action)

                # if render is required, only render worker-1
                if self.render and self.name.split('-')[1] == '1':
                    self.env.render()

                self.episode_reward += self.reward

                self.save_experience()

                self.learn()  # learn when an episode stops

                self.state = self.next_state

            # save policy
            # todo: use running reward
            if self.episode_reward >= -165:
                run_info = ''
                if self.run_num != '':
                    run_info = f'{self.run_num}_'
                name = f'{self.__class__.__name__}_solve_{self.env_id}_{run_info}{self.current_episode_num}.pt'
                torch.save(self.actor.state_dict(), os.path.join(self.policy_path, name))

            # push episode reward to global
            self.episode_rewards[self.current_episode_num - 1] = self.episode_reward
            self.length[self.current_episode_num - 1] = length
            start_index = max(self.current_episode_num - self.window, 0)
            running_reward = np.mean(self.episode_rewards[start_index: self.current_episode_num])
            self.running_rewards[self.current_episode_num - 1] = running_reward
            # print the result of this episode
            print(f'\r{self.name: <12s} work on episode {self.current_episode_num}, '
                  f'reward: {self.episode_reward: >6.1f}, running reward: {running_reward: >6.1f}', end='')

    def save_experience(self):
        """during the procedure of training, store trajectory for future learning"""
        self.reward = (self.reward + 8.1) / 8.1
        experience = (self.state, self.action, self.reward, self.state_value, self.log_prob)
        self.replayMemory.add(experience)

    def learn(self):

        # only start to learn when episode stops or collected a trajectory of given length
        if not (self.done or len(self.replayMemory) == 5):
            return

        def update(local_net, local_loss, global_net, global_optimizer):
            global_optimizer.zero_grad()
            local_loss.backward()
            for local_para, global_para in zip(local_net.parameters(), global_net.parameters()):
                global_para._grad = local_para.grad
            global_optimizer.step()

        states, actions, rewards, state_values, log_probs, advantages, _ = self.replayMemory.fetch()

        if self.done:
            value = 0
        else:
            next_state = torch.tensor(self.next_state).float().unsqueeze(0)
            value = self.critic(next_state).squeeze().detach().numpy()

        discount_rewards = discount_sum(rewards.squeeze(dim=1).numpy(), 0.99, value=value)
        # discount_rewards = torch.from_numpy(np.vstack(self.discount_rewards)).float()
        discount_rewards = torch.tensor(discount_rewards).float()

        # calculate loss for actor
        # todo: actor loss try advantage
        actor_loss = log_probs * (discount_rewards - state_values.detach())
        actor_loss = -actor_loss.mean()
        # actor_loss = -(log_probs * advantages).mean()
        update(self.actor, actor_loss, self.global_actor, self.actor_optimizer)
        self.actor.load_state_dict(self.global_actor.state_dict())

        # calculate loss for critic
        critic_loss = F.mse_loss(discount_rewards, state_values, reduction='mean')
        update(self.critic, critic_loss, self.global_critic, self.critic_optimizer)
        # self.critic = deepcopy(self.global_critic)
        self.critic.load_state_dict(self.global_critic.state_dict())


class A3C(Agent):
    def __init__(self, env, actor, critic, config):
        super(A3C, self).__init__(env, config)
        self.results_path = self.config.get('results', './results')  # path to store graph and data
        initial_folder(self.results_path, clear=self.config.get('clear_result', False))
        self.policy_path = initial_folder(self.results_path + '/policy saved', clear=False)
        self._actor = actor
        self._critic = critic
        # todo: config process num
        # self.process_num = mp.cpu_count()  # 16
        self.process_num = 8
        self.logger = setup_logger('a3c.log', name='a3cTraining')

        self.reset()  # setup some variable

        self.run = ''  # this is a flag for distinguish different run, especially when use Trainer to train multiple run

    def reset(self):
        """
        before training start, we have to initialize some variable
        """
        self.current_episode = mp.Value('i', 0)  # shared int to record the current episode number
        self.length = mp.Array('i', self.episode_num)  # record length(steps) of each episode
        self.rewards = mp.Array('d', self.episode_num)  # record reward of each episode
        self.running_rewards = mp.Array('d', self.episode_num)  # record running reward of each episode

        self.global_actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'),
                                        self.max_action)
        self.global_critic = self._critic(self.state_dim, self.config.get('critic_hidden_layer'),
                                          activation=self.config.get('critic_activation'))

        self.global_actor.init()
        self.global_critic.init()

        self.global_actor.share_memory()
        self.global_critic.share_memory()

        # todo: is share optim necessary
        self.actor_optimizer = SharedAdam(self.global_actor.parameters(), lr=1e-4, betas=(0.95, 0.999))
        self.critic_optimizer = SharedAdam(self.global_critic.parameters(), lr=1e-4, betas=(0.95, 0.999))
        # self.actor_optimizer = Adam(self.global_actor.parameters(), lr=self.config.get('learning_rate', 1e-3))
        # self.critic_optimizer = Adam(self.global_critic.parameters(), lr=self.config.get('learning_rate', 1e-3))
        self._time = time.time()

    def train(self):
        render = True if self.config.get('render') in ['train', 'both'] else False
        workers = [A3CWorker(self.env_id,
                             self.global_actor, self.global_critic, self.actor_optimizer, self.critic_optimizer,
                             self.current_episode, self.episode_num, self.length,
                             self.rewards, self.running_rewards, self.window,
                             n + 1, self.config.get('discount_factor'), render, self.run, self.policy_path)
                   for n in range(self.process_num)]

        [worker.start() for worker in workers]

        [worker.join() for worker in workers]

        # determine whether the agent has solved the problem
        episode = np.argmax(np.array(self.running_rewards) >= self.goal)
        # use np.argmax because it stops at the first True(more efficient)
        # but it might return 0 if no there is no True, so another condition is needed
        if episode > 0 or (episode == 0 and self.running_rewards[0] >= self.goal):
            print(f'\n{Color.SUCCESS}Problem solved on episode {episode + 1}, ', end='')
        else:
            print(f'\n{Color.FAIL}Problem NOT solved, ', end='')

        # calculate running time and total steps of this run
        delta_time = time.time() - self._time
        print(f'time taken: {str(datetime.timedelta(seconds=int(delta_time)))}, '
              f'total steps: {sum(self.length[:])}{Color.END}')

        # plot training result of this run
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')

        x = np.arange(1, self.episode_num + 1)
        ax.plot(x, self.rewards[:], label='reward', color=Color.REWARD, zorder=1)
        ax.hlines(y=self.goal, xmin=1, xmax=self.episode_num, label='goal', colors=Color.GOAL, zorder=2)
        ax.plot(x, self.running_rewards, label='running reward', color=Color.RUNNING_REWARD, zorder=3)

        ax.legend(loc='lower right')
        name = f'result of {self.__class__.__name__} solving {self.env_id}'
        if self.run != '':
            name += f' ({self.run}th run)'
        ax.set_title(name)
        plt.savefig(os.path.join(self.results_path, name))
        plt.close(fig)

    def load_policy(self, file):
        """load the parameter saved to value-network or policy-network for testing"""
        if self.global_actor is None:
            self.global_actor = self._actor(self.state_dim, self.action_dim, self.config.get('actor_hidden_layer'))
        self.global_actor.load_state_dict(torch.load(file))
        self.global_actor.eval()

    def test_action(self, state):
        """get the action according to the network in test scenario"""
        state = torch.tensor(state).float().unsqueeze(0)
        action_dist = self.global_actor(state)
        return action_dist.sample().detach().numpy()

    # NOTE that all the interactions with the env are performed by worker
    # i.e. the `worker` should consider how to select action, how to learn, how to save policy
    # so there is no need to implement these three methods
    def select_action(self):
        pass

    def learn(self):
        pass

    def save_policy(self):
        pass
