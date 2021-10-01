############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some constants used in the program
############################################

import matplotlib.pyplot as plt


class Color:
    """color used for printing and plotting"""
    # color for print to console
    END = '\033[0m'
    SUCCESS = '\033[94m'
    FAIL = '\033[91m'
    INFO = '\033[95m'
    WARNING = '\033[93m'

    cmap = plt.get_cmap('tab20')
    # cmap is a function which can take a value from 0 to 1 and map it to RGBA color

    # color for different agent
    DQN = cmap(0 / 20)
    DDQN = cmap(1 / 20)
    DDQN_PER = cmap(2 / 20)
    REINFORCE = cmap(3 / 20)
    ActorCritic = cmap(4 / 20)
    DDPG = cmap(5 / 20)
    TD3 = cmap(6 / 20)

    # color for different line
    cmap = plt.get_cmap('tab10')
    REWARD = cmap(0 / 10)
    TEST = cmap(0 / 10)
    GOAL = cmap(3 / 10)
    RUNNING_REWARD = cmap(1 / 10)


def visualize_color():
    """visualize all the color in COLOR"""
    x = range(1, 11)
    fig, ax = plt.subplots()
    colors = ['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'ActorCritic',
              'DDPG', 'TD3', 'GOAL', 'REWARD', 'TEST', 'RUNNING_REWARD']
    for index, color in enumerate(colors):
        print(getattr(Color, color))
        ax.plot(x, [index] * 10, color=getattr(Color, color), label=color)
    ax.legend()
    plt.show()


def get_base_config():
    return {
        # global random seed, default: None
        'seed': None,

        # basic setting of run time and episode number
        'run_num': 5,
        'episode_num': 1000,

        # parameters for replay memory
        'memory_capacity': 20000,
        'batch_size': 256,

        'discount_factor': 0.99,  # reward discount factor

        # parameters for prioritized replay memory
        'alpha': 0.3,
        'beta': 0.3,

        # folder to save results and policy
        'results': './results',
        'policy': './policy',

        # whether clear the folder of results and policy saved
        'clear_result': False,
        'clear_policy': False,

        'learning_rate': 0.01,  # start learning rate
        'learning_rate_decay_rate': 0.99,
        'min_learning_rate': 0.0001,

        'epsilon': 1,  # start epsilon
        'epsilon_decay_rate': 0.99,
        'min_epsilon': 0.01,

        'tau': 0.2,  # parameter for network soft-update

        # interval of update target_network
        'Q_update_interval': 10,  # if not specified, update every step, i.e. equals 1

        # ##### parameters for TD3 ##### #
        # specify mean and std for normal distribution noise
        'noise_mean': 0,
        # note that the value below will be multiplies with max action of the env
        'noise_std': 1,
        'noise_clip': 1,  # this value will also be multiplied with min action
        'noise_factor': 0.2,  # const to scale the noise generated by normal distribution
    }


# default goal for some env whose goal is None
DefaultGoal = {
    'Pendulum-v0': -100,
    'Humanoid‐v3': None,
    'HumanoidStandup‐v2': None,
    'Walker2d‐v3': None,
}

if __name__ == '__main__':
    visualize_color()
