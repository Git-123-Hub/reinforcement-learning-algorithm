############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some constants used in the program
############################################

from enum import Enum

DEFAULT = {
    'run_num': 3,
    'episode_num': 3,
    'seed': None,
    'resultsPath': './results',
    'policyPath': './policy',
}


class Color(Enum):
    """color used for printing and plotting"""
    # color for print
    END = '\033[0m'
    SUCCESS = '\033[94m'
    FAIL = '\033[91m'
    INFO = '\033[95m'
