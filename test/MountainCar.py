############################################
# @Author: Git-123-Hub
# @Date: 2021/9/13
# @Description: test performance of rl algorithm on problem MountainCart-v0
############################################
import gym

from utils.const import Config
from utils.model import QNet
from utils.util import compare_results
from value_based import DDQN, DDQN_PER, DQN


class ModifyReward(gym.Wrapper):
    def __init__(self, old_env):
        super(ModifyReward, self).__init__(old_env)
        self.env = old_env
        self.env.spec.reward_threshold = -80
        # note that this `new goal` is empirical,
        # only to make sure that once the agent reaches this goal, it can really learn a good policy

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # if the agent reach the goal, return more reward
        if next_state[0] >= self.env.goal_position:  # more reward if the agent reaches the goal(0.5)
            reward += 30
        elif next_state[0] >= 0.4:  # encourage the agent go further
            reward += 0.4
        elif next_state[0] >= 0.3:
            reward += 0.3
        elif next_state[0] >= 0.2:
            reward += 0.2
        elif next_state[0] >= 0.1:
            reward += 0.1
        elif next_state[0] >= -0.5:
            reward += (next_state[0] + 0.5) * 0.1

        return next_state, reward, done, info


if __name__ == '__main__':
    # NOTE that you can still solve the original env, however, with reward modified, it will become easier
    env = ModifyReward(gym.make('MountainCar-v0'))
    # env = gym.make('MountainCar-v0')

    config = Config()
    config.result_path = './Mountain_results'

    config['seed'] = 4357436
    config['run_num'] = 5
    config['episode_num'] = 1000

    config['memory_capacity'] = 1e5
    config['batch_size'] = 512

    config['learning_rate'] = 5e-4

    # value based agent
    config['Q_update_interval'] = 5
    config['tau'] = 0.01

    config['epsilon_decay_rate'] = 0.99
    config['min_epsilon'] = 0.01

    config['q_hidden_layer'] = [256, 256]
    agent = DQN(env, QNet, config)
    agent.train()

    agent = DDQN(env, QNet, config)
    agent.train()

    agent = DDQN_PER(env, QNet, config)
    agent.train()

    # todo: policy based can't solve this problem
    # # policy based
    # config['render'] = 'train'
    # config['episode_num'] = 1000 * 50
    # config['learning_rate'] = 1e-3
    # config['actor_hidden_layer'] = [128, 64]
    # a = REINFORCE(env, DiscreteStochasticActor, config)
    # a.train()

    # config['critic_hidden_layer'] = [128, 64]
    # agent = REINFORCE_BASELINE(env, DiscreteStochasticActor, StateCritic, config)
    # agent.train()

    compare_results(config.result_path)
