############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some constants used in the program
############################################

import matplotlib.pyplot as plt
from torch import nn


class Color:
    """color used for printing and plotting"""
    # color for print to console
    END = '\033[0m'
    SUCCESS = '\033[94m'
    FAIL = '\033[91m'
    INFO = '\033[95m'
    WARNING = '\033[93m'

    cmap = plt.get_cmap('tab10')
    # cmap is a function which can take a value from 0 to 1 and map it to RGBA color

    # color for different agent
    DQN = cmap(0 / 10)
    DDQN = cmap(1 / 10)
    DDQN_PER = cmap(2 / 10)
    REINFORCE = cmap(3 / 10)
    REINFORCE_BASELINE = cmap(4 / 10)
    DDPG = cmap(5 / 10)
    TD3 = cmap(6 / 10)
    SAC = cmap(7 / 10)
    PPO = cmap(8 / 10)

    # color for different line
    REWARD = cmap(0 / 10)
    TEST = cmap(0 / 10)
    GOAL = cmap(3 / 10)
    RUNNING_REWARD = cmap(1 / 10)


def visualize_color():
    """visualize all the color in COLOR"""
    x = range(1, 11)
    fig, ax = plt.subplots()
    colors = ['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'REINFORCE_BASELINE',
              'DDPG', 'TD3', 'GOAL', 'REWARD', 'TEST', 'RUNNING_REWARD']
    for index, color in enumerate(colors):
        print(getattr(Color, color))
        ax.plot(x, [index] * 10, color=getattr(Color, color), label=color)
    ax.legend()
    plt.show()


def get_base_config():
    return {
        # general config
        'seed': None,  # global random seed, default: None
        'results': './results',
        'policy': './policy',  # folder to save results and policy
        'clear_result': False,
        'clear_policy': False,  # whether clear the folder of results and policy saved or not
        'run_num': 5,
        'episode_num': 1000,  # basic setting of run time and episode number
        'discount_factor': 0.99,  # reward discount factor, i.e. gamma

        # render mode: determine whether render or not,
        # 'train' for rendering during training, 'test' for rendering during testing policy,
        # 'both' for render in both scenario, default 'None', do not render
        'render': None,

        'memory_capacity': 20000,
        'batch_size': 256,  # parameters for replay memory
        'alpha': 0.3,
        'beta': 0.3,  # parameters for prioritized replay memory

        # todo: maybe add learning rate to different network
        'learning_rate': 1e-3,  # learning rate

        'epsilon': 1,  # start epsilon
        'epsilon_decay_rate': 0.99,
        'min_epsilon': 0.01,

        'tau': 0.01,  # parameter for network soft-update

        'Q_update_interval': 5,  # interval of update target_network, if not specified, update every step, i.e. equals 1

        # ##### parameters for TD3 ##### #
        # specify mean and std for normal distribution noise
        'noise_mean': 0,
        # note that the value below will be multiplies with max action of the env
        'noise_std': 1,
        'noise_clip': 1,  # this value will also be multiplied with min action
        'noise_factor': 0.2,  # const to scale the noise generated by normal distribution

        # ##### SAC ##### #
        'entropy_coefficient': None,  # use alpha auto tuning, or you can specify a number

        # ##### hidden layer for different network ##### #
        'q_hidden_layer': [128, 128],
        'policy_hidden_layer': [128, 128],
        'actor_hidden_layer': [128, 128],
        'critic_hidden_layer': [128, 128],
        # activation of network
        'q_activation': nn.ReLU(),
        'policy_activation': nn.ReLU(),
        'actor_activation': nn.ReLU(),
        'critic_activation': nn.ReLU(),

        # ##### parameters for PPO ##### #
        'training_epoch': 50,
        'clip_ratio': 0.2,
    }


# default goal for some env whose goal is None
DefaultGoal = {
    'Pendulum-v0': -165,
    'Humanoid‐v3': None,
    'HumanoidStandup‐v2': None,
    'Walker2d‐v3': None,
}

if __name__ == '__main__':
    visualize_color()
