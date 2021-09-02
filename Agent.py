############################################
# @Author: 123
# @Date: 2021/8/31
# @Description: abstract class for all RL agent
############################################
import datetime
import os
import pickle
import random
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from const import DEFAULT, Color
from util import setup_logger, initial_folder


class Agent:
    """
    Abstract class for all RL agent. Wrap gym.Env with some data structure to record the performance of the algorithm.
    """

    def __init__(self, env: gym.Env, config: dict):
        self.env = env
        self.env_id = self.env.unwrapped.spec.id
        self.goal = self.env.spec.reward_threshold
        # todo: should i add trail

        # get the basic information about the `env`
        self.state_dim = env.observation_space.shape[0]
        # todo: some env don't have the following property
        # self.action_dim = env.action_space.shape[0]
        # self.max_action = env.action_space.high[0]
        # self.min_action = env.action_space.low[0]

        self.config = config

        self.logger = setup_logger(self.__class__.__name__ + '_training.log', name='training_data_logger')

        # path to store graph and data from the training
        self.results_path = './results'
        self.policy_path = './saved policy'

        self.results_path = initial_folder(self.config.get('results', DEFAULT['resultsPath']))
        self.policy_path = initial_folder(self.config.get('policy', DEFAULT['policyPath']))

        self.run_num = self.config.get('run_num', DEFAULT['run_num'])  # total run num
        self.episode_num = self.config.get('episode_num', DEFAULT['episode_num'])  # episode num of each run

        # todo: specify a number !!!!! how to increase
        self._run, self._episode = None, None
        # current run num and episode num during training
        # And it will plus 1 when print to the user considering of the readability

        self.length = np.zeros((self.run_num, self.episode_num), dtype=np.int)
        # record the length of every episode, index from 0
        # {self.length[n][m]} represents the length of the (n+1)th run, (m+1)th episode

        self.rewards = np.zeros((self.run_num, self.episode_num), dtype=np.float64)
        # record total reward of every episode, index from 0
        # {self.rewards[n][m]} represents the reward of the (n+1)th run, (m+1)th episode

        self._time = None
        # record the time of every run.
        # record start time in {self.run_reset()}, print program running time after every run.

        # todo: rolling results, max action, min action

        # get random seed for each run: generate a list of random seeds using the seed specified in config
        np.random.seed(self.config.get('seed', DEFAULT['seed']))
        self._seeds = np.random.randint(0, 2 ** 32 - 2, size=self.run_num, dtype=np.int64)
        # random seed of current run, updated in `self.run_reset`
        self._seed = None

    def set_random_seed(self, *, more_random: bool = False):
        """
        set random seed of the current run of the training, so that the results can be reproduced
        :param more_random: specify this keyword argument as `True` to set seed `None` and make the random more `random`
        """
        # We take the seed provided in `self.config` as the global seed,
        # and use it the generate random seeds for different run (self._seeds).
        # Or you can specify `more_random` as True to ignore the seed,
        # and make the procedure more random, e.g. when testing the policy.

        if not more_random:  # get the seed and make the training deterministic
            seed = self._seeds[self._run].item()  # convert numpy.int64 to int
            torch.manual_seed(seed)  # seed passed to torch.manual_seed() can't be None, so we call it here
        else:
            seed = None

        # global seed setting
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        # torch setting:
        torch.backends.cudnn.deterministic = not more_random
        torch.backends.cudnn.benchmark = more_random
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)

        # gym setting
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(seed)
        self.env.action_space.seed(seed)
        # set the seed of env.action_space so that action_space.sample() can be deterministic
        # or you can choose to use another random function(e.g. randint) to implement how to choose action randomly
        return seed

    def run_reset(self):
        """reset the agent before each run"""
        # set random seed before anything to make sure the algorithm can be reproduced
        self._seed = self.set_random_seed()
        # record the start time of this run
        self._time = time.time()
        print(self.__class__.__name__ + f' solves {self.env_id} {self._run + 1}th run, random seed: {self._seed}')

    def episode_reset(self):
        """reset the agent to start another episode"""
        # todo: use rolling_result to update learning rate
        # self.update_learning_rate()

        # set seed before env.reset(), to make sure the initial state after reset() can be the same
        # if you don't need it, comment the following line
        # todo: if the env.seed behave like this
        self.env.seed(self._seed)
        # Note that env.reset() would also reset the seed for env
        self.state = self.env.reset()
        self.action, self.next_state, self.reward, self.done = None, None, None, False

        self.logger.info(f'{self._run + 1}th run, {self._episode + 1}th episode start at state {self.state}')

    def train(self, run_num=None, episode_num=None):
        if run_num is None: run_num = self.run_num
        if episode_num is None: episode_num = self.episode_num
        for run in range(run_num):
            self._run = run
            self.run_reset()
            for episode in range(episode_num):
                self._episode = episode
                self.episode_reset()
                self.run_an_episode()
            # calculate the time of this run
            self._time = time.time() - self._time
            print(f' time taken: {str(datetime.timedelta(seconds=int(self._time)))}{Color.END}')
        self.save_results()
        print(f'{Color.INFO}Training Finished.{Color.END}\n')

    def run_an_episode(self):
        raise NotImplementedError

    def save_results(self):
        """save training data and figure after training finishes"""
        self.plot('rewards')
        self.plot('length')
        self.save_data()

    def plot(self, data_name):
        """plot data to visualize the performance of the algorithm"""
        # the data to be plotted should be a ndarray of size(self.run_num, self.episode_num)
        # you can also define other data structure to inspect performance for each run, each episode
        if data_name not in {'rewards', 'length'}:
            raise ValueError("data_name should be 'rewards' or 'length'")
        x = np.arange(1, self.episode_num + 1)
        data = eval('self.' + data_name)

        fig, ax = plt.subplots()
        ax.set_xlabel('episode num')
        ax.set_ylabel(data_name)
        for run in range(self.run_num):  # plot all the data in one figure
            ax.plot(x, data[run], label=f'{run + 1}th run')
        ax.legend(loc='upper left')
        name = f'{data_name} of {self.__class__.__name__} solving {self.env_id}'
        ax.set_title(name)
        plt.savefig(os.path.join(self.results_path, name))
        fig.clear()

    def save_data(self):
        """save training data after training stops"""
        name = self.__class__.__name__ + '_training_data.pkl'
        path = os.path.join(self.results_path, name)
        # todo maybe just rewards is enough,
        #  because other variables can be calculated from it, or save some other variables.
        # todo: add some other variable: some training info, time, data, training time
        data = {
            'rewards': self.rewards,
            # 'rolling_result': self.rolling_results,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    # todo: should test_policy in the base Agent
