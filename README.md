# reinforcement-learning-algorithm

- `object oriented`: all of the rl agents using the same framework(base class `Agent`), makes it easy to read and
  understand
- `perfect reproduction`: training result would be exactly the same under the same random seed

# algorithm implemented and corresponding paper

- DQN
  [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602),
  [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236?wm=book_wap_0005)
- DDQN
  [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- Dueling DQN
  [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- DDQN with prioritized experience replay
  [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- REINFORCE(Monte-Carlo Policy Gradient, Vanilla Policy Gradient)
- REINFORCE with BASELINE
- DDPG
  [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- TD3
  [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- SAC
  [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  ,
  [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- PPO [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- A3C [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

# online result

> training result of the agent trying to solve a problem from a scratch

### CartPole-v1

![CartPole-v1](results/CartPole-v1-online.png)

### MountainCar-v0
> the original environment is hard to converge, <br>
> so I modify the reward to solve this problem and get the result below

![MountainCar-v0](results/MountainCar-v0-online.png)

### LunarLander-v2

![LunarLander-v2](results/LunarLander-v2-online.png)

### Acrobot-v1

![Acrobot-v1](results/Acrobot-v1-online.png)

### Pendulum-v0
> Note that there is no goal for Pendulum-v0,
> but as you can see in the result,
> the agent did learn something

![Pendulum-v0](results/Pendulum-v0-online.png)

### HalfCheetah-v3

![HalfCheetah-v3](results/HalfCheetah-v3-online.png)

# offline result

> online training is not always stable<br>
> sometimes the agent gets a high reward(or running reward)<br>
> then its performance would decline rapidly.<br>
> so I choose some policy during the training to test the agent's performance

### CartPole-v1

![CartPole-v1](results/CartPole-v1-offline.png)
![CartPole-v1-visualize](results/CartPole-v1.gif)

### MountainCar-v0

> though training on a modified environment, <br>
> I still use the original one to test the policy, <br>
> thus illustrate the result the learning

![MountainCar-v0](results/MountainCar-v0-offline.png)
![MountainCar-v0-visualize](results/MountainCar-v0.gif)

### LunarLander-v2

![LunarLander-v2](results/LunarLander-v2-offline.png)
![LunarLander-v2-visualize](results/LunarLander-v2.gif)

### Acrobot-v1

![Acrobot-v1](results/Acrobot-v1-offline.png)
![Acrobot-v1-visualize](results/Acrobot-v1.gif)

### Pendulum-v0

[comment]: <> (![Pendulum-v0]&#40;results/Acrobot-v1-offline.png&#41;)
![Pendulum-v0-visualize](results/Pendulum-v0.gif)

### HalfCheetah-v3

![HalfCheetah-v3](results/HalfCheetah-v3-offline.png)
![HalfCheetah-v3-visualize](results/HalfCheetah-v3.gif)

# inspired by

- [Deep Reinforcement Learning Algorithms with PyTorch By p-christ](
  https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
- [reinforcement learning examples of pytorch](
  https://github.com/pytorch/examples/tree/master/reinforcement_learning)
- [Author's PyTorch implementation of TD3](https://github.com/sfujim/TD3)
- [RLkit](https://github.com/rail-berkeley/rlkit)
- [Spinning-Up](https://spinningup.openai.com/en/latest/)
- [pytorch-A3C by MorvanZhou](https://github.com/MorvanZhou/pytorch-A3C)
