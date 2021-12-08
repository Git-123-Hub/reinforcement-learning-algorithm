############################################
# @Author: Git-123-Hub
# @Date: 2021/12/6
# @Description: test all the base functionality of all the agents
############################################
import gym

from policy_based import TD3, A3C, DDPG, PPO, REINFORCE, REINFORCE_BASELINE, SAC
from utils.const import Config
from utils.model import QNet, DeterministicActor, StateActionCritic, StateCritic, ContinuousStochasticActor, \
    DiscreteStochasticActor
from value_based import DDQN_PER, DDQN, DQN

if __name__ == '__main__':
    config = Config()
    config.run_num = 2
    config.episode_num = 10
    # test discrete env with CartPole
    env = gym.make('CartPole-v1')

    config.result_path = './test_results'

    for agent in [DQN, DDQN, DDQN_PER]:
        agent(env, QNet, config).train()
    REINFORCE(env, DiscreteStochasticActor, config).train()
    REINFORCE_BASELINE(env, DiscreteStochasticActor, StateCritic, config).train()

    # test continuous env with Pendulum
    env = gym.make('Pendulum-v0')
    DDPG(env, DeterministicActor, StateActionCritic, config).train()
    TD3(env, DeterministicActor, StateActionCritic, config).train()
    PPO(env, ContinuousStochasticActor, StateCritic, config).train()
    A3C(env, ContinuousStochasticActor, StateCritic, config).train()

    env = gym.make('HalfCheetah-v3')
    SAC(env, ContinuousStochasticActor, StateActionCritic, config)
