############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some constants used in the program
############################################

from enum import Enum

DEFAULT = {
    'run_num': 3,
    'episode_num': 250,
    'seed': None,
    'resultsPath': './results',
    'policyPath': './policy',
    'learning_rate': 0.01,
    'epsilon': 0.1,
    'clip_grad': None,
}


class Color:
    """color used for printing and plotting"""
    # color for print
    END = '\033[0m'
    SUCCESS = '\033[94m'
    FAIL = '\033[91m'
    INFO = '\033[95m'
