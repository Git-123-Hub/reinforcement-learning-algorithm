############################################
# @Author: Git-123-Hub
# @Date: 2021/8/31
# @Description: get basic information about env in gym
############################################

from gym import envs
from pprint import pprint

import gym

# get all the env in gym
# print(envs.registry.all())

env_list = [
    # classic control
    'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0',
    # box2D
    'BipedalWalker-v3', 'CarRacing-v0', 'BipedalWalkerHardcore-v3',
    'LunarLander-v2', 'LunarLanderContinuous-v2',
    # mujoco
    'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2',
    'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v3', 'Walker2d-v3']
info = {}
for env_name in env_list:
    env = gym.make(env_name)
    _info = {}
    actionSpaceName = env.action_space.__class__.__name__
    if actionSpaceName == 'Discrete':
        _info['actionMode'] = 'discrete'
        _info['action_dim'] = env.action_space.n
    elif actionSpaceName == 'Box':
        _info['actionMode'] = 'continuous'
        _info['action_dim'] = env.action_space.shape[0]
        _info['max_action'] = env.action_space.high
        _info['min_action'] = env.action_space.low
    else:
        print(f'unnoticed action space type: ${actionSpaceName}')

    _info['goal'] = env.spec.reward_threshold

    state_dim = env.observation_space.shape[0]
    _info['state_dim'] = state_dim
    info[env_name] = _info

pprint(info)
