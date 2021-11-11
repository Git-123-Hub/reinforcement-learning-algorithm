############################################
# @Author: Git-123-Hub
# @Date: 2021/11/10
# @Description: implementation of A3C(Asynchronous Advantage Actor Critic)
############################################
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils import Agent, EpisodicReplayMemory
import torch.multiprocessing as mp
from utils.util import setup_logger


class A3CWorker(mp.Process):
    """A3C worker that interact with environment asynchronously for an episode"""

    def __init__(self, env, global_actor, global_critic, global_episode_num, global_episode_rewards, total_episode_num,
                 actor_optimizer, critic_optimizer):
        super(A3CWorker, self).__init__()
        # setup env
        self.env = deepcopy(env)
        self.state, self.action, self.reward, self.next_state, self.done = None, None, None, None, False
        self.state_value, self.log_prob = None, None
        # these two value should also be stored in replay memory for learning
        self.episode_reward = 0
        # reward of the current episode, this value will be pushed to `self.global_episode_rewards`
        # when episode terminates, and will finally be used to plot reward with respect to episode num

        # get state_dim and action_dim for initializing the network
        self.state_dim = env.observation_space.shape[0]
        if env.action_space.__class__.__name__ == 'Discrete':
            self.action_dim = env.action_space.n
        elif env.action_space.__class__.__name__ == 'Box':  # continuous
            self.action_dim = env.action_space.shape[0]
            self.max_action = self.env.action_space.high[0]
            self.min_action = self.env.action_space.low[0]

        # setup network
        self.global_actor = deepcopy(global_actor)
        self.global_critic = deepcopy(global_critic)
        self.actor, self.critic = None, None

        # self.actor_optimizer = actor_optimizer
        # self.critic_optimizer = critic_optimizer
        self.actor_optimizer = Adam(global_actor.parameters(), lr=1e-3)
        self.critic_optimizer = Adam(global_critic.parameters(), lr=1e-3)
        # NOTE that these optimizers are for global net
        # local net doesn't need to be optimized, it just copy from global

        self.global_episode_num = global_episode_num
        self.global_episode_rewards = global_episode_rewards
        self.total_episode_num = total_episode_num
        self.current_episode_num = 0

        # todo: add config
        self.replayMemory = EpisodicReplayMemory(0.99)

        self.logger = setup_logger(self.__class__.__name__ + '_training.log', name='training_data')

    def episode_reset(self):
        """operation before an episode starts"""
        self.current_episode_num = self.global_episode_num.value
        # we also maintain `current_episode_num` so that when an episode stops, we can get `current` episode accurately
        # not the global value of that moment, which might have been changed by other workers

        self.replayMemory.reset()
        # copy from the global network
        self.actor = deepcopy(self.global_actor)
        self.critic = deepcopy(self.global_critic)

        self.actor_optimizer = Adam(self.global_actor.parameters(), lr=1e-3)
        self.critic_optimizer = Adam(self.global_critic.parameters(), lr=1e-3)

        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0  # reset reward of the current episode

    def select_action(self):
        # with torch.no_grad():
        state = torch.tensor(self.state).float().unsqueeze(0)
        action_dist = self.actor(state)
        self.action = action_dist.sample()
        self.log_prob = action_dist.log_prob(self.action).sum()
        self.action = self.action.detach().numpy()
        self.state_value = self.critic(state).squeeze()  # shape: torch.Size([]), just a tensor

    def run(self):
        """
        this is the function that will be executed when `worker.start()` is invoked
        in our situation, it should be the training method
        """
        while self.global_episode_num.value < self.total_episode_num:
            self.global_episode_num.value += 1
            self.episode_reset()

            # agent interact with env for an episode
            while not self.done:
                self.select_action()
                # execute action
                self.next_state, self.reward, self.done, _ = self.env.step(self.action)
                # todo: only render one worker
                # todo: maybe also record length

                self.episode_reward += self.reward

                self.save_experience()
                self.state = self.next_state

            self.learn()  # learn when an episode stops
            # todo: how to calculate running rewards
            # push episode reward to global
            self.global_episode_rewards[self.current_episode_num - 1] = self.episode_reward
            # print the result of this episode
            print(f'{self.name} work on episode {self.current_episode_num}, '
                  f'reward: {self.episode_reward}')

    def save_experience(self):
        """during the procedure of training, store trajectory for future learning"""
        experience = (self.state, self.action, self.reward, self.state_value, self.log_prob)
        self.replayMemory.add(experience)

    def learn(self):

        def update(local_net, local_loss, global_net, global_optimizer):
            global_optimizer.zero_grad()
            local_loss.backward()
            for local_para, global_para in zip(local_net.parameters(), global_net.parameters()):
                global_para._grad = local_para.grad
            global_optimizer.step()

        states, actions, rewards, state_values, log_probs, advantages, discount_rewards = self.replayMemory.fetch()

        # calculate loss for actor
        # todo: actor loss try advantage
        actor_loss = log_probs * (discount_rewards - state_values.detach())
        update(self.actor, actor_loss.mean(), self.global_actor, self.actor_optimizer)

        # calculate loss for critic
        critic_loss = F.mse_loss(discount_rewards, state_values, reduction='mean')
        update(self.critic, critic_loss, self.global_critic, self.critic_optimizer)


class A3C:
    def __init__(self, env, actor, critic, config):
        # todo: how to record each episode's performance and each run's performance
        self.env = env
        self.config = config
        self.global_actor = actor(3, 1, self.config.get('actor_hidden_layer'))
        self.global_critic = critic(3, self.config.get('critic_hidden_layer'),
                                    activation=self.config.get('critic_activation'))
        # print(self.global_actor)
        # print(self.global_critic)

        self.global_actor.share_memory()
        self.global_critic.share_memory()

        self.config = config

        self.actor_optimizer = Adam(self.global_actor.parameters(), lr=self.config.get('learning_rate', 1e-3))
        self.critic_optimizer = Adam(self.global_critic.parameters(), lr=self.config.get('learning_rate', 1e-3))

        self.episode_num = self.config.get('episode_num', 1000)
        # self.episode_num = 30
        self.episode_rewards = mp.Array('d', self.episode_num)  # a shared array to record reward of each episode
        # todo: add current run
        self.current_episode = mp.Value('i', 0)  # shared int to record the current episode number

        # todo: config process num
        # self.process_num = mp.cpu_count()  # 16
        self.process_num = 8
        workers = [A3CWorker(env, self.global_actor, self.global_critic, self.current_episode, self.episode_rewards,
                             self.episode_num, self.actor_optimizer, self.critic_optimizer)
                   for _ in range(self.process_num)]
        [worker.start() for worker in workers]

        [worker.join() for worker in workers]

        # print(self.episode_rewards[0:5])
        # print(self.episode_rewards[5:10])
        # print(self.episode_rewards[10:15])
        # print(self.episode_rewards[15:20])
        # print(self.episode_rewards[20:25])
        # print(self.episode_rewards[25:30])
        fig, ax = plt.subplots()
        ax.plot(range(len(self.episode_rewards[:])), self.episode_rewards[:])
        plt.savefig('a3c.png')
        plt.close(fig)
