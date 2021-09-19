############################################
# @Author: Git-123-Hub
# @Date: 2021/8/31
# @Description: get basic information about env in gym
############################################

# get all the env in gym
# from gym import envs
# print(envs.registry.all())
from pprint import pprint

import gym

envs = ['CartPole-v1', 'MountainCarContinuous-v0',
        'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v0']
info = {}
for env_name in envs:
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
