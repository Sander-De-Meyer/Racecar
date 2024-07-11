import gym
from stable_baselines3 import PPO

# Create the environment
env = CarRacingEnv('/mnt/data/track2.png')

# Define the model
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_car_racing")

# Load the model
model = PPO.load("ppo_car_racing")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    env.render()
