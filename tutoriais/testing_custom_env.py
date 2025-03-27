import gym_pybullet_drones
import gymnasium as gym
import numpy as np
import random

env = gym.make("GPS-Distance-v0", rng=random.Random(42), gui=True)

print("-"*64)
# print(env.observation_space.shape[0])
print("Action Space:")
print(env.action_space)  # Discrete(4) <class 'gymnasium.spaces.discrete.Discrete'>
print(env.action_space.shape[1])
print("Observation Space:")
print(env.observation_space)
observation, info = env.reset(seed=42)
print("-"*64)
print("Observation:")
print(observation)
print("info:")
print(info)
print("-"*64)

action = env.action_space.sample()  # this is where you would insert your policy
print("Action", action)
observation, reward, terminated, truncated, info = env.step(action)
print(observation[0].shape)
print(observation[1].shape)
print(observation[2].shape)
new_observation = np.concatenate(
   (observation[0].flatten(), 
    observation[1].flatten(), 
    observation[2].flatten()), axis=0)
print(new_observation.shape)
# print("Observation:", observation)
# print("Reward:", reward)
# print("Terminated:", terminated)
# print("Truncated:", truncated)
# print("Info:", info)

print("------------")

if terminated or truncated:
   observation, info = env.reset()

for _ in range(10):
   action = env.action_space.sample()  # this is where you would insert your policy
   print("Action", action)
   observation, reward, terminated, truncated, info = env.step(action)

   print("Observation:", observation)
   print("Reward:", reward)
   print("Terminated:", terminated)
   print("Truncated:", truncated)
   print("Info:", info)

   print("------------")

#    if terminated or truncated:
#       observation, info = env.reset()

env.close()