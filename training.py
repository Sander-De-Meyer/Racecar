import gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from DQN_Agent import DQN_Agent
from environment import CartPole
from car import Car

print("starting with environment parameters")
# Set environment and training parameters
num_episodes_train = 140 #200
num_episodes_test = 10 # 20
learning_rate = 5e-4

# Create the environment
env = gym.make('CartPole-v0')
env = CartPole()
env = Car()

action_space_size = env.action_space.n
state_space_size = 4

# Plot average performance of 5 trials
num_seeds = 4 # 5
l = num_episodes_train // 10
res = np.zeros((num_seeds, l))
gamma = 0.99



print("Start looping")
# Loop over multiple seeds
for i in tqdm.tqdm(range(num_seeds)):
    reward_means = []

    # Create an instance of the DQN_Agent class
    agent = DQN_Agent(env, lr=learning_rate)

    # Training loop
    for m in range(num_episodes_train):
        agent.train()

        # Evaluate the agent every 10 episodes during training
        if m % 10 == 0:
            print("Episode: {}".format(m))

            # Evaluate the agent's performance over 20 test episodes
            G = np.zeros(num_episodes_test)
            for k in range(num_episodes_test):
                g = agent.test()
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            print(f"The test reward for episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")
            reward_means.append(reward_mean)

    res[i] = np.array(reward_means)

print("Start plotting")

# Plotting the average performance
ks = np.arange(l) * 10
avs = np.mean(res, axis=0)
maxs = np.max(res, axis=0)
mins = np.min(res, axis=0)

plt.fill_between(ks, mins, maxs, alpha=0.1)
plt.plot(ks, avs, '-o', markersize=1)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Avg. Return', fontsize=15)
plt.show()