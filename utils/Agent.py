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
import abc

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.util import setup_logger, initial_folder
from utils.const import Color, DefaultGoal, Config


# NOTE that this is a Abstract Class for agent, a subclass will have to implement following method:
# select_action(), learn(), save_policy(), load_policy(), test_action()

class Agent(abc.ABC):
    """
    Abstract class for all RL agent.
    Wrap gym.Env with some data structure to record the performance of the algorithm.
    """

    def __init__(self, env: gym.Env, config: Config):
        # set up environment and get some base information about this environment
        self.env = env
        self.env_id = self.env.unwrapped.spec.id
        self.goal = self.env.spec.reward_threshold
        if self.goal is None: self.goal = DefaultGoal[self.env_id]
        # todo: remove these variable to train()
        self.state, self.action, self.next_state, self.reward, self.done = None, None, None, None, False

        # get state_dim and action_dim for initializing the network
        self.state_dim = env.observation_space.shape[0]
        if env.action_space.__class__.__name__ == 'Discrete':
            self.action_dim = env.action_space.n
        elif env.action_space.__class__.__name__ == 'Box':  # continuous
            self.action_dim = env.action_space.shape[0]
            self.max_action = self.env.action_space.high[0]
            self.min_action = self.env.action_space.low[0]

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # some algorithms might need a replay buffer to store experience, if needed, just initialize it
        # related methods have already been implemented: reset() in `run_reset()`,  save_experience()
        self.replay_buffer = None

        self.logger = setup_logger(self.__class__.__name__ + '_training.log', name='training_data')

        # path to store some data
        self.result_path = initial_folder(self.config.result_path)  # path to store graph and data
        self.policy_path = initial_folder(self.config.result_path + '/policy saved')  # path to store network parameters

        self.run_num = self.config.run_num  # total run num
        self.episode_num = self.config.episode_num  # episode num of each run

        self._run, self._episode = None, None
        # current run_num and episode_num during training
        # it will plus 1 when print to the user considering of the readability

        self.length = np.zeros((self.run_num, self.episode_num), dtype=int)
        # record the length of every episode, index from 0
        # `self.length[n][m]` represents the length of the (n+1)th run, (m+1)th episode
        # NOTE that `self.length[n].sum()` represents total steps of `n+1`th run

        self.rewards = np.zeros((self.run_num, self.episode_num), dtype=np.float64)
        # record total reward of every episode, index from 0
        # `self.rewards[n][m]` represents the reward of the (n+1)th run, (m+1)th episode

        # todo: different env might have different moving window
        self.window = 100
        self.running_rewards = np.zeros((self.run_num, self.episode_num), dtype=np.float64)
        # average reward of the last `self.window` episodes, can be used to measure the algorithm's stability
        # Note that according to openai.gym, the criteria for success(agent solves the problem) is:
        # the average reward of `self.window` consecutive episodes greater than `self.goal`
        # i.e. `self.running_rewards` reaches `self.goal`

        self._time = None
        # record the time of every run.
        # record start time in `self.run_reset()`, print program running time after every run.

        self.set_random_seed()  # set up seed before training starts

    def set_random_seed(self, *, more_random: bool = False):
        """
        set random seed of the current run, so that the results can be reproduced
        :param more_random: specify this keyword argument as `True` to set seed `None` and make the random more `random`
        """
        if self.config.seed is None:
            return
        if not more_random:  # get the seed and make the training deterministic
            seed = self.config.seed  # global seed specified in `config`

            # seed for functions below can't be None, so we call it here
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
        if hasattr(gym.spaces, 'prng'): gym.spaces.prng.seed(seed)
        self.env.seed(seed)
        # set the seed of env.action_space so that action_space.sample() can be deterministic
        # or you can choose to use another random function(e.g. randint) to implement how to choose action randomly
        self.env.action_space.seed(seed)
        print(f'seed has been set to {seed}')

    def run_reset(self):
        """reset the agent before each run"""
        if self.replay_buffer is not None: self.replay_buffer.reset()

        # record the start time of this run
        self._time = time.time()
        print(self.__class__.__name__ + f' solves {self.env_id} {self._run + 1}th run')

    def episode_reset(self):
        """reset the agent to start another episode"""
        self.state = self.env.reset()
        self.action, self.next_state, self.reward, self.done = None, None, None, False

        self.logger.info(f'{self._run + 1}th run, {self._episode + 1}th episode start at state {self.state}')

    def train(self):
        for run in range(self.run_num):
            self._run = run
            self.run_reset()
            for episode in range(self.episode_num):
                self._episode = episode
                self.episode_reset()
                self.run_an_episode()
                self.save_policy()

            self.plot_run_result()

            # run finishes, print the result of this run
            # determine whether the agent has solved the problem
            episode = np.argmax(self.running_rewards[self._run] >= self.goal)
            # use np.argmax because it stops at the first True(more efficient)
            # but it might return 0 if no there is no True, so another condition is needed
            if episode > 0 or (episode == 0 and self.running_rewards[self._run][0] >= self.goal):
                print(f'\n{Color.SUCCESS}Problem solved on episode {episode + 1}, ', end='')
            else:
                print(f'\n{Color.FAIL}Problem NOT solved, ', end='')

            # calculate running time and total steps of this run
            delta_time = time.time() - self._time
            print(f'time taken: {str(datetime.timedelta(seconds=int(delta_time)))}, '
                  f'total steps: {self.length[self._run].sum()}{Color.END}')

        # training finishes, plot result and save data
        self.plot_running_rewards()
        # save the instance of the Agent to keep all the training data
        name = self.__class__.__name__ + '_training_data.pkl'
        path = os.path.join(self.result_path, name)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print(f'{Color.INFO}Training Finished.{Color.END}')

    def run_an_episode(self):
        """procedure of the agent interact with the env for a whole episode"""
        while not self.done:
            self.select_action()
            # execute action
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            if self.config.render in ['train', 'both']: self.env.render()

            self.length[self._run][self._episode] += 1
            self.rewards[self._run][self._episode] += self.reward

            self.save_experience()
            self.learn()

            self.state = self.next_state

        self.running_rewards[self._run][self._episode] = self._running_reward

        # episode finishes, print the result of this episode
        print(f'\repisode: {self._episode + 1: >3}, '
              f'steps: {self.length[self._run][self._episode]: >4}, '
              f'reward: {self.rewards[self._run][self._episode]: >5.1f}, '
              f'running reward: {self._running_reward: >5.3f}', end='')

    @abc.abstractmethod
    def select_action(self):
        """
        determine how the agent choose action given `self.state`,
        consider it might choose according to value-network or policy-network
        this method varies in different algorithms, so it should be implemented by subclass
        """
        raise NotImplementedError

    def save_experience(self):
        """
        some algorithms need to store experience for sampling, while others don't
        so experience is saved only if `self.replayMemory` has already been defined
        """
        if self.replay_buffer is not None:
            self.replay_buffer.add(self.state, self.action, self.reward, self.next_state, self.done)

    @abc.abstractmethod
    def learn(self):
        """
        this is where the algorithm itself is implemented
        specify the learning behavior of the agent on every step or every episode(when to learn and how to learn)
        obviously, this method should be implemented in subclass
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_policy(self):
        """save the parameters of the current network"""
        # Note that you can save the policy in a fixed step interval to evaluate the agent
        # here, I only consider saving policy when `self.running_reward` reaches `self.goal`
        # different algorithm might need to save different network
        # so this method should be implemented in subclass
        raise NotImplementedError

    def plot_run_result(self):
        """plot reward, running reward, goal at the same figure after each run"""
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')
        x = np.arange(1, self.episode_num + 1)
        ax.plot(x, self.rewards[self._run], label='reward', color=Color.REWARD, zorder=1)
        ax.hlines(y=self.goal, xmin=1, xmax=self.episode_num, label='goal', colors=Color.GOAL, zorder=2)
        ax.plot(x, self.running_rewards[self._run], label='running reward', color=Color.RUNNING_REWARD, zorder=3)

        ax.legend(loc='lower right')
        name = f'result of {self.__class__.__name__} solving {self.env_id}({self._run + 1}th run)'
        ax.set_title(name)
        plt.savefig(os.path.join(self.result_path, name))
        plt.close(fig)

    def plot_running_rewards(self):
        """plot `self.running_rewards` statistically to indicate the performance(stability) of the training"""
        # initialize a figure
        fig, ax = plt.subplots()
        ax.set_xlabel('episode')
        ax.set_ylabel('running reward')

        # prepare data
        x = np.arange(1, self.episode_num + 1)
        mean = np.mean(self.running_rewards, axis=0)
        std = np.std(self.running_rewards, axis=0)
        color = getattr(Color, self.__class__.__name__)  # different color for different agent

        ax.plot(x, mean, color=color)
        ax.plot(x, mean - std, color=color, alpha=0.2)
        ax.plot(x, mean + std, color=color, alpha=0.2)
        ax.fill_between(x, y1=mean - std, y2=mean + std, color=color, alpha=0.1)
        ax.hlines(y=self.goal, xmin=1, xmax=self.episode_num, label='goal', color=Color.GOAL)

        # for run in range(self.run_num):  # plot running reward of each run separately
        #     ax.plot(x, self.running_rewards[run], label=f'{run + 1}th run')
        # ax.legend(loc='lower right')

        name = f'running reward of {self.__class__.__name__} solving {self.env_id}'
        ax.set_title(name)
        plt.savefig(os.path.join(self.result_path, name))
        plt.close(fig)

    @property
    def _running_reward(self):
        """calculate running reward of the current episode,
        i.e. the average reward of the last `self.window` episodes"""
        return np.mean(self.rewards[self._run][max(self._episode + 1 - self.window, 0):self._episode + 1])

    def test(self, n: int = None, *, episodes=None):
        """
        test the performance of the policy(or value-network) saved during training
        i.e. the parameter file saved by `self.save_policy()`
        :type n: int, if specified, then choose `n` policies randomly from `self.policy_path`,
         otherwise test all the policies
        :type episodes: int, the number of episode that each policy will be tested
        """
        # Note that this is a framework of the testing
        # a subclass need to implement method `load_policy()` and `test_action()` to make it work

        if episodes is None:
            episodes = self.window
        # episodes should be greater than `self.window`, otherwise, it's meaningless
        elif episodes < self.window:
            episodes = self.window
            print(f"{Color.WARNING}Warning that the test episodes has been set to {self.window}, "
                  f"because it's too small{Color.END}")

        # first, we set random more `random` to test strictly
        self.set_random_seed(more_random=True)

        # prepare all the policies that will be tested
        policies = os.listdir(self.policy_path)
        if len(policies) == 0:
            print(f'{Color.FAIL}No policy found{Color.END}')
            return
        if n is not None: policies = random.sample(policies, n)

        # initialize a figure
        fig, ax = plt.subplots()
        x = np.arange(1, episodes + 1)

        for file_name in policies:
            file = os.path.join(self.policy_path, file_name)
            # set up the network with the parameters we need to test
            self.load_policy(file)

            # define some variable to record performance of the testing
            rewards = np.zeros(episodes)
            running_rewards = np.zeros(episodes)

            # remake the env just in case that the training environment is different from the official one
            # for example: modified with wrapper
            env = gym.make(self.env_id)
            goal = env.spec.reward_threshold
            if goal is None: goal = DefaultGoal[self.env_id]

            for episode in range(episodes):  # test for given episodes
                env.seed()
                state = env.reset()
                done = False

                while not done:  # test for an episode
                    if self.config.render in ['test', 'both']: env.render()
                    # time.sleep(0.01)

                    action = self.test_action(state)
                    next_state, reward, done, _ = env.step(action)
                    rewards[episode] += reward
                    state = next_state

                # print training result of this episode
                running_rewards[episode] = np.mean(rewards[max(episode - self.window + 1, 0):episode + 1])
                print(f'\rTesting policy {file_name}: episode: {episode + 1}, '
                      f'reward: {rewards[episode]}, '
                      f'running reward: {running_rewards[episode]: .2f}', end=' ')

            # testing of this policy file is finished, evaluate the performance of the testing
            # running rewards only make sense when the agent runs at least `self.window` episodes
            if np.any(running_rewards[self.window - 1:] >= goal):
                print(f'{Color.SUCCESS}Test Passed{Color.END}')
            else:
                print(f'{Color.FAIL}Test Failed{Color.END}')
                # consider there might be a lot of policies saved, you can delete a policy if it fails the test
                # or you may comment the line below to not delete it
                os.remove(file)

            # plot the test result
            ax.set_xlabel('episode')
            ax.set_ylabel('rewards')
            ax.plot(x, rewards, label='test', color=Color.TEST)
            ax.plot(x, running_rewards, label='running reward', color=Color.RUNNING_REWARD)
            ax.hlines(y=goal, xmin=1, xmax=episodes, label='goal', color=Color.GOAL)
            ax.legend(loc='lower right')
            name = f'{os.path.splitext(file_name)[0]} test results'  # get filename without extension
            ax.set_title(name)
            plt.savefig(os.path.join(self.result_path, name))
            ax.clear()

        plt.close(fig)
        print(f'{Color.INFO}Test Finished{Color.END}')

    @abc.abstractmethod
    def load_policy(self, file):
        """load the parameter saved to value-network or policy-network for testing"""
        # parameters might be loaded to different network according to different algorithm
        # so this method should be implemented by subclass
        raise NotImplementedError

    @abc.abstractmethod
    def test_action(self, state):
        """get the action according to the network in test scenario"""
        # because the network might behave different(the output is unknown)
        # and different algorithm choose action in a different way
        # so this method should be implemented by subclass
        raise NotImplementedError
