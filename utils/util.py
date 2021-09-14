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
import torch

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


def soft_update(from_network, to_network, tau):
    """
    copy the parameters of `from_network` to `to_network` with a proportion of tau
    i.e. update `to_network` parameter with tau of `from_network`
    """
    for to_p, from_p in zip(to_network.parameters(), from_network.parameters()):
        to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)


def compare(agents, path):
    """
    compare the results of `agents` using the training data saved during training
    :param agents: a list of agent name
    :param path: folder where training data is stored
    :return: a figure showing statistic results of agents
    """
    env_id = None
    # initialize a figure for the data to plot
    fig, ax = plt.subplots()
    ax.set_xlabel('episode')
    ax.set_ylabel('running rewards')
    for agent_name in agents:
        file = os.path.join(path, f'{agent_name}_training_data.pkl')
        with open(file, 'rb') as f:
            agent = pickle.load(f)

            # make sure that these data are solving the same problem
            if env_id is None:
                env_id = agent.env_id
            else:
                assert agent.env_id == env_id

            # prepare data
            x = np.arange(1, agent.running_rewards.shape[1] + 1)  # get episode length
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
    # convert `agents`(a list of string) to a string for better look
    names = ''.join(f'{name} ' for name in agents)
    name = f'statistical running rewards of {names} solving {env_id}'
    ax.set_title(name)
    ax.legend(loc='upper left')
    plt.savefig(os.path.join(path, name))
    fig.clear()


def transfer_experience(experiences):
    """
    transfer experience sampled form replay memory into mini-batches that can be passed to network
    i.e. transfer a list of experiences where each element is an experience in the form of
    (state, action, reward, next_states, done), into four parts where all the states store in an array
    all the actions store in an array...
    :param experiences: a list of experiences
    """
    # first stack the sampled experiences vertically,
    experiences = np.vstack(experiences).transpose()
    # so that each row represents a tuple for experience, and each row represents same component(s,a,r,d,s')
    # after transpose, states are transferred to the first row, actions on the second row...

    # transfer array to tensor to pass the the network
    # Note that the first dimension has to be batch size, so i use np.vstack before convert to tensor
    states = torch.from_numpy(np.vstack(experiences[0])).float()
    actions = torch.from_numpy(np.vstack(experiences[1])).float()
    rewards = torch.from_numpy(np.vstack(experiences[2])).float()
    next_states = torch.from_numpy(np.vstack(experiences[3])).float()
    dones = torch.from_numpy(np.vstack(experiences[4])).float()
    return states, actions, rewards, next_states, dones
