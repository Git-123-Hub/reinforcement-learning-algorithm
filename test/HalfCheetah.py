############################################
# @Author: Git-123-Hub
# @Date: 2021/9/28
# @Description: test the performance of RL algorithm on problem HalfCheetah-v3
############################################
import gym
from torch import nn

from policy_based import DDPG, TD3, SAC, PPO
from utils.const import get_base_config
from utils.model import DeterministicActor, StateActionCritic, ContinuousStochasticActor, StateCritic, \
    ContinuousStochasticActorFixStd

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v3')
    config = Config()
    config.result_path = './HalfCheetah_results'
    # config['seed'] = 482307631
    config['run_num'] = 3
    config['episode_num'] = 1000

    config['memory_capacity'] = 1e5
    config['batch_size'] = 256

    config['Q_update_interval'] = 1
    config['tau'] = 0.005

    config['learning_rate'] = 3e-4

    config['actor_hidden_layer'] = [256, 256]
    config['critic_hidden_layer'] = [256, 256]
    # it seems that DDPG can't solve this problem, but it can learn something
    agent = DDPG(env, DeterministicActor, StateActionCritic, config)
    agent.train()
    # agent.test()

    config['update_interval'] = 2
    config['noise_std'] = 1
    config['noise_clip'] = 0.5
    config['noise_factor'] = 0.2
    agent = TD3(env, DeterministicActor, StateActionCritic, config)
    agent.train()
    # agent.test(render=True)

    # config['seed'] = 4823231
    config['learning_rate'] = 1e-3
    config['actor_hidden_layer'] = [256, 256]
    agent = SAC(env, ContinuousStochasticActor, StateActionCritic, config)
    agent.train()

    # ! this PPO can learn something, but not good enough to solve this problem
    config['critic_activation'] = nn.Tanh()
    config['learning_rate'] = 5e-4
    config['training_epoch'] = 10  # todo: try 10
    config['episode_num'] = 500 * 60
    # config['render'] = 'train'
    print(config['training_epoch'], config['episode_num'])
    agent = PPO(env, ContinuousStochasticActorFixStd, StateCritic, config)
    # agent = PPO(env, ContinuousStochasticActor, StateCritic, config)
    agent.train()
