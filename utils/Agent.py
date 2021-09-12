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
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.util import setup_logger, initial_folder
from utils.const import Color


class Agent:
    """
    Abstract class for all RL agent. Wrap gym.Env with some data structure to record the performance of the algorithm.
    """

    def __init__(self, env: gym.Env, config: dict):
        self.env = env
        self.env_id = self.env.unwrapped.spec.id
        self.goal = self.env.spec.reward_threshold

        # get the basic information about the `env`
        self.state_dim = env.observation_space.shape[0]
        # todo: some env don't have the following property
        # self.action_dim = env.action_space.shape[0]
        # self.max_action = env.action_space.high[0]
        # self.min_action = env.action_space.low[0]

        self.config = config

        self.logger = setup_logger(self.__class__.__name__ + '_training.log', name='training_data_logger')

        # path to store graph and data from the training
        self.results_path = self.config.get('results', './results')
        initial_folder(self.results_path, clear=self.config.get('clear_result', False))
        self.policy_path = self.config.get('policy', './policy')
        initial_folder(self.policy_path, clear=self.config.get('clear_policy', False))

        self.run_num = self.config.get('run_num', 3)  # total run num
        self.episode_num = self.config.get('episode_num', 250)  # episode num of each run

        self._run, self._episode = None, None
        # current run num and episode num during training
        # And it will plus 1 when print to the user considering of the readability

        self.length = np.zeros((self.run_num, self.episode_num), dtype=int)
        # record the length of every episode, index from 0
        # {self.length[n][m]} represents the length of the (n+1)th run, (m+1)th episode

        self.rewards = np.zeros((self.run_num, self.episode_num), dtype=np.float64)
        # record total reward of every episode, index from 0
        # {self.rewards[n][m]} represents the reward of the (n+1)th run, (m+1)th episode

        # todo: different env might have different moving window
        self.window = 100
        self.running_rewards = np.zeros((self.run_num, self.episode_num), dtype=np.float64)
        # average of the last `self.window` episodes' rewards, can be used to measure the algorithm's stability
        # Note that according to openai.gym, the criteria for success(agent solves the problem) is:
        # `self.running_rewards` reaches `self.goal`

        self._time = None
        # record the time of every run.
        # record start time in {self.run_reset()}, print program running time after every run.

        self._global_seed = self.config.get('seed')
        # the 'seed' specified in config is the global seed of the training
        # if provided, we use it to generate a list of random seeds used in each run
        if self._global_seed is not None:
            np.random.seed(self._global_seed)
            self._seeds = np.random.randint(0, 2 ** 32 - 2, size=self.run_num, dtype=np.int64)
        # if not provided, then each run of the training use 'None'

        # random seed of current run, default None
        # if `self._global_seed` is provided, it will be updated in `self.run_reset`
        self._seed = None

        self.optimizer = None

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

            # seeds for function below can't be None, so we call it here
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
        else:
            seed = None

        # global seed setting
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        # torch setting:
        torch.backends.cudnn.deterministic = not more_random
        torch.backends.cudnn.benchmark = more_random

        # gym setting
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(seed)
        self.env.action_space.seed(seed)
        # set the seed of env.action_space so that action_space.sample() can be deterministic
        # or you can choose to use another random function(e.g. randint) to implement how to choose action randomly
        return seed

    def run_reset(self):
        """reset the agent before each run"""
        # if `self._global_seed` is not provided, there is no need to set random seed
        if self._global_seed is not None:
            # if provided, set random seed before anything to make sure the algorithm can be reproduced
            self._seed = self.set_random_seed()

        # record the start time of this run
        self._time = time.time()
        print(self.__class__.__name__ + f' solves {self.env_id} {self._run + 1}th run, random seed: {self._seed}')

    def episode_reset(self):
        """reset the agent to start another episode"""
        self.update_learning_rate()

        # set seed before env.reset(), to make sure the initial state after reset() can be the same
        # so that the result of the algorithm under the same random seed can be exactly same
        self.env.seed(self._seed)
        # however, start from an unique initial state,
        # the agent might achieve high reward simply by remembering sequences of actions
        # if you don't need it, comment the following line

        self.state = self.env.reset()
        # Note that env.reset() would also reset the seed for env
        # and that is the reason why we have to set seed before reset()

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
                self.running_rewards[self._run][self._episode] = self._running_reward
                self.save_policy()

            # print info about this episode's training
            # determine whether the agent has solved the problem
            episode = np.argmax(self.running_rewards[self._run] >= self.goal)
            # use np.argmax because it stops at the first True(more efficient)
            # but it might return 0 if no there is no True, so another condition is needed
            if episode > 0 or (episode == 0 and self.running_rewards[self._run][0] >= self.goal):
                print(f'\n{Color.SUCCESS}Problem solved on episode {episode + 1}, ', end=' ')
            else:
                print(f'\n{Color.FAIL}Problem NOT solved, ', end=' ')
            # calculate running time and total steps of this run
            self._time = time.time() - self._time
            print(f'time taken: {str(datetime.timedelta(seconds=int(self._time)))}, '
                  f'total steps: {self.length.sum()}{Color.END}')

        self.save_results()
        print(f'{Color.INFO}Training Finished.{Color.END}\n')

    def run_an_episode(self):
        """procedure of the agent interact with the env for a whole episode"""
        raise NotImplementedError

    def save_policy(self):
        """save the parameters of the current policy(or Q network)"""
        # Note that you can save the policy in a fixed step interval to evaluate the agent
        # here, I only consider save policy when the running reward reaches `self.goal`
        raise NotImplementedError

    def save_results(self):
        """save training data and figure after training finishes"""
        self.plot('rewards')
        self.plot('running_rewards')
        self.plot('length')
        self.save()

    def plot(self, data_name):
        """plot data to visualize the performance of the algorithm"""
        # the data to be plotted should be a ndarray of size(self.run_num, self.episode_num)
        # you can also define other data structure to inspect performance for each run, each episode
        if data_name not in {'rewards', 'length', 'running_rewards'}:
            raise ValueError("data_name should be 'rewards', 'length' or 'running_rewards'")
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

    def save(self):
        """save the instance of this class after training stops"""
        name = self.__class__.__name__ + '_training_data.pkl'
        path = os.path.join(self.results_path, name)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def _running_reward(self):
        """calculate the average of the last `self.window` episodes' rewards"""
        return np.mean(self.rewards[self._run][max(self._episode + 1 - self.window, 0):self._episode + 1])

    def update_learning_rate(self):
        """
        update learning rate of optimizer according to last episode's reward,
        as the reward get closer to `self.goal`, learning rate should become lower
        """

        # todo: maybe max reward is also a good criteria
        def _update_in_optimizer(lr):
            for p in self.optimizer.param_groups: p['lr'] = lr

        start_lr = self.config['learning_rate']
        if self._episode == 0:  # first episode of new run, reset learning rate
            self._learning_rate = start_lr
            _update_in_optimizer(start_lr)
            self.logger.info(f'learning rate reset to {start_lr}')
            return
        old_lr = self._learning_rate
        pre_running_reward = self.running_rewards[self._run][self._episode - 1]
        if pre_running_reward <= 0.25 * self.goal:
            self._learning_rate = start_lr
        elif pre_running_reward <= 0.5 * self.goal:
            self._learning_rate = start_lr / 2
        elif pre_running_reward <= 0.6 * self.goal:
            self._learning_rate = start_lr / 10
        elif pre_running_reward <= 0.75 * self.goal:
            self._learning_rate = start_lr / 20
        else:
            self._learning_rate = start_lr / 100

        if self._learning_rate != old_lr:
            _update_in_optimizer(self._learning_rate)
            self.logger.info(f'change learning rate to {self._learning_rate}')

    def test(self, n: int = None, *, episodes=None):
        """
        test the performance of the policy(or Q network) saved during training
        i.e. the parameter file saved by `self.save_policy()`
        :type n: int, if specified, then choose `n` policies randomly from `self.policy_path`.
        if not provided, test all the policies
        :type episodes: int, the number of episode that each policy will be tested
        """
        if episodes is None:
            episodes = self.window
        # episodes should be greater than `self.window`, otherwise, it's meaningless
        elif episodes < self.window:
            episodes = self.window
            print(f"{Color.WARNING}Warning that the test episodes has been set to {self.window}, "
                  f"because it's too small{Color.END}")
        # first, we set random more `random`
        self.set_random_seed(more_random=True)
        all_policies = os.listdir(self.policy_path)
        if len(all_policies) == 0:
            print(f'{Color.FAIL}No policy found{Color.END}')
            return
        policies = random.sample(all_policies, n)
        for file_name in policies:
            self._evaluate_policy(file_name, episodes)

        print(f'{Color.INFO}Test Finished{Color.END}')

    def _evaluate_policy(self, file_name, episodes):
        raise NotImplementedError
