############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some constants used in the program
############################################

class Color:
    """color used for printing and plotting"""
    # color for print
    END = '\033[0m'
    SUCCESS = '\033[94m'
    FAIL = '\033[91m'
    INFO = '\033[95m'
    WARNING = '\033[93m'

    # color for different agent
    DQN = 'blue'
    DDQN = 'green'
    DDQN_PER = 'yellow'
    REINFORCE = 'purple'
    ActorCritic = 'black'


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
    }


# default goal for some env whose goal is None
DefaultGoal = {
    'Pendulum-v0': -100,
    'Humanoid‐v3': None,
    'HumanoidStandup‐v2': None,
    'Walker2d‐v3': None,
}
