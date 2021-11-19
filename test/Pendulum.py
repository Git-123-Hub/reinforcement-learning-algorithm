############################################
# @Author: Git-123-Hub
# @Date: 2021/9/23
# @Description: solve problem Pendulum using rl algorithm
############################################
import gym

from policy_based import DDPG, TD3, PPO, A3C
from utils import Trainer
from utils.const import get_base_config
from utils.model import DeterministicActor, StateActionCritic, ContinuousStochasticActor, StateCritic, \
    ContinuousStochasticActorFixStd

if __name__ == '__main__':
    # NOTE that there is no goal for Pendulum-v0, but as you can see in the result, the agent did learn something
    env = gym.make('Pendulum-v0')
    config = get_base_config()
    config['results'] = './Pendulum_results'
    config['policy'] = './Pendulum_policy'
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

    config['actor_hidden_layer'] = [128, 64]
    config['critic_hidden_layer'] = [128, 64]
    config['learning_rate'] = 3e-4

    config['training_epoch'] = 5
    config['episode_num'] = 300 * 30
    # agent = PPO(env, ContinuousStochasticActor, StateCritic, config)
    agent = PPO(env, ContinuousStochasticActorFixStd, StateCritic, config)
    # agent.train()

    config['episode_num'] = 30
    config['actor_hidden_layer'] = [128, 128]
    config['critic_hidden_layer'] = [128, 128]
    config['seed'] = 123
    agent = A3C(env, ContinuousStochasticActorFixStd, StateCritic, config)
    # agent.train()
    trainer = Trainer(agent, 3)
    trainer.train()
    # agent.test()
