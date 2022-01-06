############################################
# @Author: Git-123-Hub
# @Date: 2021/9/5
# @Description: test RL agent's performance on solving problem LunarLander
############################################
import gym

from policy_based import REINFORCE, REINFORCE_BASELINE
from utils.const import Config
from utils.model import QNet, DiscreteStochasticActor, StateCritic
from utils.util import compare_results
from value_based import DDQN, DDQN_PER, DQN, DuelingQNet

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    config = Config()
    config.result_path = './LunarLander_results'

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
    # NOTE that these method can't solve the problem, it can learn something, but can't reach the goal
    config['learning_rate'] = 5e-3
    config['episode_num'] = 1000
    config['actor_hidden_layer'] = [64, 64]
    agent = REINFORCE(env, DiscreteStochasticActor, config)
    agent.train()

    config['critic_hidden_layer'] = [64, 64]
    agent = REINFORCE_BASELINE(env, DiscreteStochasticActor, StateCritic, config)
    agent.train()

    compare_results(config.result_path)
