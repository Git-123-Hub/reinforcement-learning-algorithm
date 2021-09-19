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

        # get state_dim and action_dim for initializing the network
        self.state_dim = env.observation_space.shape[0]
        if env.action_space.__class__.__name__ == 'Discrete':
            self.action_dim = env.action_space.n
        elif env.action_space.__class__.__name__ == 'Box':
            self.action_dim = env.action_space.shape[0]

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

            self.plot_run_result()

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

    def plot_run_result(self):
        """plot reward, running reward, goal at the same figure"""
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')
        x = np.arange(1, self.episode_num + 1)
        ax.plot(x, self.rewards[self._run], label='reward')
        ax.plot(x, self.running_rewards[self._run], label='running reward')
        ax.hlines(y=self.goal, xmin=1, xmax=self.episode_num, colors='red', label='goal')
        ax.legend(loc='upper left')
        name = f'result of {self.__class__.__name__} solving {self.env_id}({self._run + 1}th run)'
        ax.set_title(name)
        plt.savefig(os.path.join(self.results_path, name))
        fig.clear()

    def save_results(self):
        """save training data and figure after training finishes"""
        # todo: may be these plot method can be removed
        # self.plot('rewards')
        self.plot('running_rewards')
        # self.plot('length')
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
        start_lr = self.config['learning_rate']
        min_lr = self.config.get('min_learning_rate', 0.0001)
        if self._episode == 0:  # first episode of new run, reset learning rate
            self._learning_rate = start_lr
            self.logger.info(f'learning rate reset to {start_lr}')
        elif self.running_rewards[self._run][self._episode - 1] >= self.goal:
            self._learning_rate = min_lr
        else:
            self._learning_rate *= self.config.get('learning_rate_decay_rate', 0.99)
            self._learning_rate = max(self._learning_rate, min_lr)

        # update learning rate in optimizer
        for p in self.optimizer.param_groups: p['lr'] = self._learning_rate

    def test(self, n: int = None, *, episodes=None, render=False):
        """
        test the performance of the policy(or Q network) saved during training
        i.e. the parameter file saved by `self.save_policy()`
        :type n: int, if specified, then choose `n` policies randomly from `self.policy_path`.
        if not provided, test all the policies
        :type episodes: int, the number of episode that each policy will be tested
        """
        # Note that a subclass need to inherit method `load_policy()` and `test_action()`

        if episodes is None:
            episodes = self.window
        # episodes should be greater than `self.window`, otherwise, it's meaningless
        elif episodes < self.window:
            episodes = self.window
            print(f"{Color.WARNING}Warning that the test episodes has been set to {self.window}, "
                  f"because it's too small{Color.END}")

        # first, we set random more `random`
        self.set_random_seed(more_random=True)

        policies = os.listdir(self.policy_path)
        if len(policies) == 0:
            print(f'{Color.FAIL}No policy found{Color.END}')
            return

        if n is not None:
            policies = random.sample(policies, n)

        # plot the test result of this policy
        fig, ax = plt.subplots()

        for file_name in policies:
            file = os.path.join(self.policy_path, file_name)
            # load the parameters saved for testing
            # the network might be Q or policy, so this method should be inherited by subclass
            self.load_policy(file)

            # define some variable to record performance
            rewards = np.zeros(episodes)
            running_rewards = np.zeros(episodes)
            # remake the env just in case that the training environment is different from the official one
            env = gym.make(self.env_id)
            goal = env.spec.reward_threshold
            for episode in range(episodes):  # test for given episodes
                env.seed()
                state = env.reset()
                done = False

                while not done:  # test for an episode
                    # you can uncomment the following two lines to visualize the procedure and not to fast
                    if render:
                        env.render()
                    # time.sleep(0.01)

                    # get action according to the network
                    # because the network might behave different(the output is unknown)
                    # so this method should also be inherited by subclass
                    action = self.test_action(state)
                    next_state, reward, done, _ = env.step(action)
                    rewards[episode] += reward
                    state = next_state

                running_rewards[episode] = np.mean(rewards[max(episode - self.window + 1, 0):episode + 1])
                print(f'\rTesting policy {file_name}: episode: {episode + 1}, '
                      f'reward: {rewards[episode]}, '
                      f'running reward: {format(running_rewards[episode], ".2f")}', end=' ')

            # evaluate the performance of the testing
            # running rewards only make sense when the agent runs at least `self.window` episodes
            if np.any(running_rewards[self.window - 1:] >= goal):
                print(f'{Color.SUCCESS}Test Passed{Color.END}')
            else:
                print(f'{Color.FAIL}Test Failed{Color.END}')
                # consider there might be a lot of policies saved
                # you can delete a policy if it fails the test
                # or you may comment the line below to not to delete it
                os.remove(file)

            ax.set_xlabel('episode')
            ax.set_ylabel('rewards')
            name = f'{os.path.splitext(file_name)[0]} test results'  # get filename without extension
            ax.set_title(name)
            ax.plot(np.arange(1, episodes + 1), rewards, label='test')
            ax.plot(np.arange(1, episodes + 1), running_rewards, label='running rewards')
            ax.hlines(y=goal, xmin=1, xmax=episodes, label='goal', alpha=0.5)
            ax.legend(loc='upper left')
            plt.savefig(os.path.join(self.results_path, name))
            ax.clear()

        print(f'{Color.INFO}Test Finished{Color.END}')

    def load_policy(self, file):
        """load the parameter saved to Q network or policy network"""
        raise NotImplementedError

    def test_action(self, state):
        """get the action in test scenario"""
        raise NotImplementedError
