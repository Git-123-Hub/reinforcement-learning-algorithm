############################################
# @Author: 123
# @Date: 2021/8/31
# @Description: get basic information about env in gym
############################################

"""
get some basic information about the env: discrete or continuous, state dimension, action dimension, max action...etc
and plot them in a table
"""

# get all the env in gym
# from gym import envs
# print(envs.registry.all())
import gym
import matplotlib.pyplot as plt
import numpy as np


# todo: env_id: env.unwrapped.spec.id, goal:env.spec.reward_threshold
envs = ['CartPole-v0', 'CartPole-v1', 'MountainCarContinuous-v0']
info = {}
for env_name in envs:
    env = gym.make(env_name)
    _info = {}
    actionSpaceName = env.action_space.__class__.__name__
    if actionSpaceName == 'Discrete':
        actionMode = 'discrete'
        _info['actionMode'] = actionMode
    elif actionSpaceName == 'Box':
        actionMode = 'continuous'
        _info['actionMode'] = actionMode
        _info['action_dim'] = env.action_space.shape
        _info['max_action'] = env.action_space.high
        _info['min_action'] = env.action_space.low
    else:
        print(f'unnoticed action space type: ${actionSpaceName}')

    # todo: get action meanings()
    state_dim = env.observation_space.shape[0]
    info[env_name] = _info

print(info)

fig, ax = plt.subplots(figsize=(10.0, 6.0))  # default: 8.0, 6.0
table = ax.table(cellText=np.random.random((3, 3)),
                 rowLabels=envs,
                 colLabels=['1', '2', '3', '4', '5', ], loc='center')
table.auto_set_column_width([0, 1, 2, 3])
table.auto_set_font_size()
ax.set_title('123')
ax.axis('tight')
# ax.axis('off')
# plt.subplots_adjust(left=0.2, top=0.8)
plt.savefig('envInfo.png')
