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

from const import Color


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


def initial_folder(folder, *, clear=False):
    """
    create folder if not exist, remove all the files in the folder if already exists
    :param folder: path to the folder
    :param clear: bool indicate whether clear all the file in the folder
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif clear:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    return folder


def compare(agents):
    """
    compare the results of `agents` using the training data saved during training
    :param agents: a list of agent name
    :return: a figure showing statistic results of agents
    """
    root = './results'
    env_id = None
    # initialize a figure for the data to plot
    fig, ax = plt.subplots()
    ax.set_xlabel('episode')
    ax.set_ylabel('running rewards')
    for agent_name in agents:
        file = os.path.join(root, f'{agent_name}_training_data.pkl')
        with open(file, 'rb') as f:
            data = pickle.load(f)

            # make sure that these data are solving the same problem
            if env_id is None:
                env_id = data['env_id']
            else:
                assert data['env_id'] == env_id

            mean = np.mean(data['running_reward'], axis=0)
            std = np.std(data['running_reward'], axis=0)
            x = np.arange(1, data['running_reward'].shape[1] + 1)  # get episode length
            color = getattr(Color, agent_name)
            ax.plot(x, mean, color=color, label=agent_name)
            ax.plot(x, mean - std, color=color, alpha=0.1)
            ax.plot(x, mean + std, color=color, alpha=0.1)
            ax.fill_between(x, y1=mean - std, y2=mean + std, color=color, alpha=0.1)
    # convert `agents`(a list of string) to a string for better look
    names = ''.join(f'{name} ' for name in agents)
    name = f'statistical running rewards of {names} solving {env_id}'
    ax.set_title(name)
    ax.legend(loc='upper left')
    plt.savefig(os.path.join(root, name))
    fig.clear()
