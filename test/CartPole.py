############################################
# @Author: Git-123-Hub
# @Date: 2021/9/2
# @Description: test RL agent's performance on solving problem CartPole
############################################

import gym

from policy_based import REINFORCE, REINFORCE_BASELINE
from utils.const import Config
from utils.model import QNet, DiscreteStochasticActor, StateCritic
from utils.util import compare
from value_based import DDQN, DDQN_PER, DQN

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    config = Config()

    # base config
    config.result_path = './CartPole_results'
    config['seed'] = 5326583
    config['run_num'] = 3
    config['episode_num'] = 500

    config['min_epsilon'] = 0.001
    config['Q_update_interval'] = 5
    config['tau'] = 0.05

    config['learning_rate'] = 5e-4

    config['memory_capacity'] = 10000
    config['batch_size'] = 256

    config['q_hidden_layer'] = [128]
    agent = DQN(env, QNet, config)
    agent.train()

    agent = DDQN(env, QNet, config)
    agent.train()

    agent = DDQN_PER(env, QNet, config)
    agent.train()
    # agent.test(20)

    config.episode_num = 1000
    config['learning_rate'] = 0.001
    config['policy_hidden_layer'] = [128]
    agent = REINFORCE(env, DiscreteStochasticActor, config)
    agent.train()
    # agent.test()

    config['critic_hidden_layer'] = [32, 32]
    agent = REINFORCE_BASELINE(env, DiscreteStochasticActor, StateCritic, config)
    agent.train()
    # agent.test()

    compare(['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'REINFORCE_BASELINE'], config.result_path)
