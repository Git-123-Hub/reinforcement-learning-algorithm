############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem Acrobot-v1
############################################


import gym

from policy_based import REINFORCE, REINFORCE_BASELINE
from utils.const import get_base_config
from utils.model import QNet, DiscreteStochasticActor, StateCritic
from utils.util import compare
from value_based import DDQN, DDQN_PER, DQN

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    config = get_base_config()

    # base config
    config['results'] = './Acrobot-results'
    config['policy'] = './Acrobot-policy'
    config['seed'] = 756170127
    config['run_num'] = 3
    config['episode_num'] = 500

    config['learning_rate'] = 0.001

    # config for DQN
    config['min_epsilon'] = 0.001
    config['Q_update_interval'] = 20
    config['tau'] = 0.1

    config['q_hidden_layer'] = [32, 32]
    agent = DQN(env, QNet, config)
    agent.train()

    agent = DDQN(env, QNet, config)
    agent.train()

    agent = DDQN_PER(env, QNet, config)
    agent.train()

    # NOTE that the performance of these policy based method is not stable
    # sometimes it can solve the problem, while it may also learn nothing
    config['learning_rate'] = 0.005
    config['episode_num'] = 1000
    config['policy_hidden_layer'] = [32, 32]
    agent = REINFORCE(env, DiscreteStochasticActor, config)
    agent.train()

    config['critic_hidden_layer'] = [64]
    agent = REINFORCE_BASELINE(env, DiscreteStochasticActor, StateCritic, config)
    agent.train()

    compare(['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'REINFORCE_BASELINE'], config['results'])
