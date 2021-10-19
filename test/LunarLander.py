############################################
# @Author: Git-123-Hub
# @Date: 2021/9/5
# @Description: test RL agent's performance on solving problem LunarLander
############################################
import gym

from policy_based import REINFORCE, REINFORCE_BASELINE
from utils.const import get_base_config
from utils.model import QNet, DiscreteStochasticActor, StateCritic
from utils.util import compare
from value_based import DDQN, DDQN_PER, DQN, DuelingQNet

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    config = get_base_config()
    config['results'] = './LunarLander_results'
    config['policy'] = './LunarLander_policy'

    config['seed'] = 4375233
    config['run_num'] = 3
    config['episode_num'] = 400

    # value based agent
    config['Q_update_interval'] = 4
    config['tau'] = 0.01
    config['learning_rate'] = 1e-4

    config['q_hidden_layer'] = [128, 128]
    agent = DQN(env, QNet, config)
    agent.train()

    agent = DDQN(env, QNet, config)
    agent.train()

    agent = DDQN_PER(env, QNet, config)
    agent.train()

    # policy based
    # todo: tuning hyperparameter
    # NOTE that these method can't solve the problem, it can learn something, but can't reach the goal
    config['learning_rate'] = 5e-3
    config['episode_num'] = 1000
    config['actor_hidden_layer'] = [64, 64]
    a = REINFORCE(env, DiscreteStochasticActor, config)
    a.train()

    config['critic_hidden_layer'] = [64, 64]
    agent = REINFORCE_BASELINE(env, DiscreteStochasticActor, StateCritic, config)
    agent.train()

    compare(['DQN', 'DDQN', 'DDQN_PER', 'REINFORCE', 'REINFORCE_BASELINE'], config['results'])
