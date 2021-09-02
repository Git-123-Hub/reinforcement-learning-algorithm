############################################
# @Author: Git-123-Hub
# @Date: 2021/9/1
# @Description: some useful functions
############################################

import logging
import os


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
    create folder if not exist, remove all the files in the folder if already exists
    :param folder: path to the folder
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    return folder
