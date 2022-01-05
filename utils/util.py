############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some useful functions
############################################

import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.const import Color


def setup_logger(filename, name=__name__):
    """
    set up logger with filename and logger name.
    :param filename: file to store the log data
    :param name: specify name for logger for distinguish
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(name)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def initial_folder(folder):
    """
    create folder if not exist
    :param folder: path of the folder
    :return: this created folder
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def soft_update(from_network, to_network, tau):
    """
    copy the parameters of `from_network` to `to_network` with a proportion of tau
    i.e. update `to_network` parameter with tau of `from_network`
    """
    for to_p, from_p in zip(to_network.parameters(), from_network.parameters()):
        to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)


def compare_results(path):
    """
    compare all the training data(.pkl) available in the `path` and plot them in the same figure
    :param path: folder where training data is stored
    """
    env_id = None  # name of the problem solved
    # initialize a figure
    fig, ax = plt.subplots()
    ax.set_xlabel('episode')
    ax.set_ylabel('running rewards')

    training_data = [file for file in os.listdir(path) if file.endswith('.pkl')]
    for file_name in training_data:
        file = os.path.join(path, file_name)
        with open(file, 'rb') as f:
            agent = pickle.load(f)
            agent_name = agent.__class__.__name__

            # make sure that these data are solving the same problem
            if env_id is None:
                env_id = agent.env_id
            else:
                assert agent.env_id == env_id, "data of solving different env shouldn't be plotted together"

            # prepare data
            x = np.arange(1, agent.episode_num + 1)  # get episode length
            mean = np.mean(agent.running_rewards, axis=0)
            std = np.std(agent.running_rewards, axis=0)

            color = getattr(Color, agent_name)  # different color for different agent
            label = agent_name  # label for different agent
            # if the agent used the dueling network, it should be shown in the label
            if hasattr(agent, 'dueling') and agent.dueling is True:
                label = agent_name + '-dueling'

            ax.plot(x, mean, color=color, label=label)
            ax.plot(x, mean - std, color=color, alpha=0.1)
            ax.plot(x, mean + std, color=color, alpha=0.1)
            ax.fill_between(x, y1=mean - std, y2=mean + std, color=color, alpha=0.1)

    ax.hlines(y=agent.goal, xmin=1, xmax=agent.episode_num + 1, label='goal', color=Color.GOAL)
    name = f'statistical running rewards of {env_id}'
    ax.set_title(name)
    ax.legend(loc='upper left')
    plt.savefig(os.path.join(path, name))


def discount_sum(x, gamma, *, normalize=False, value=0):
    """
    calculate the discounted cumsum of input vector x
    :param value: start value
    :param x: input vector: [x0, x1, x2]
    :param gamma: discount factor
    :param normalize: keyword argument determine whether normalize the output vector with mean and std
    :return: output vector: [x0 + gamma * x1 + gamma^2 * x2, x1 + gamma * x2, x2]
    """
    result = np.zeros_like(x, dtype=float)
    # v0 = 0
    for index in reversed(range(len(x))):
        value = x[index] + gamma * value
        result[index] = value

    if normalize:
        eps = np.finfo(np.float32).eps.item()  # tiny non-negative number
        result = (result - result.mean()) / (result.std() + eps)

    return result


if __name__ == '__main__':
    # test function discounted_sum()
    x_ = [1, 2, 3]
    gamma_ = 0.99
    y = [
        x_[0] + x_[1] * gamma_ + x_[2] * gamma_ ** 2,
        x_[1] + x_[2] * gamma_,
        x_[2]
    ]
    print(discount_sum(x_, gamma_))
    np.testing.assert_almost_equal(discount_sum(x_, gamma_), y)
