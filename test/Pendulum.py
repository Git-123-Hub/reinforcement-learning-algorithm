############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: solve problem Pendulum using rl algorithm
############################################
import gym

from policy_based import DDPG, TD3, PPO, A3C
from utils.const import Config
from utils.model import DeterministicActor, StateActionCritic, ContinuousStochasticActor, StateCritic
from utils.util import compare_results


class ModifyReward(gym.Wrapper):
    def __init__(self, old_env):
        super(ModifyReward, self).__init__(old_env)
        self.env = old_env

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = (reward + 8.1) / 8.1
        # NOTE: this reward shaping is taken from A3C implementation of MorvanZhou
        # https://github.com/MorvanZhou/pytorch-A3C/blob/master/continuous_A3C.py#L95
        return next_state, reward, done, info


if __name__ == '__main__':
    # NOTE that there is no goal for Pendulum-v0, but as you can see in the result, the agent did learn something
    env = gym.make('Pendulum-v0')
    config = Config()
    config.result_path = './Pendulum_results'
    config['seed'] = 103423575
    config['run_num'] = 3
    config['episode_num'] = 300

    config['memory_capacity'] = 10000
    config['batch_size'] = 64

    config['Q_update_interval'] = 1
    config['tau'] = 0.01

    config['learning_rate'] = 0.0005

    config['actor_hidden_layer'] = [128, 64]
    config['critic_hidden_layer'] = [128, 64]
    agent = DDPG(env, DeterministicActor, StateActionCritic, config)
    agent.train()
    # agent.test()

    config['update_interval'] = 2
    config['noise_std'] = 1
    config['noise_clip'] = 0.5
    config['noise_factor'] = 0.2

    agent = TD3(env, DeterministicActor, StateActionCritic, config)
    agent.train()
    # agent.test()

    config['learning_rate'] = 3e-4

    config['training_epoch'] = 5
    config['episode_num'] = 300 * 30
    config.fix_std = None
    agent = PPO(env, ContinuousStochasticActor, StateCritic, config)
    agent.train()

    config['episode_num'] = 3000
    config.learning_rate = 1e-4
    config['learn_interval'] = 5
    # config['process_num'] = 5
    agent = A3C(ModifyReward(env), ContinuousStochasticActor, StateCritic, config)
    agent.train()

    compare_results(config.result_path)
