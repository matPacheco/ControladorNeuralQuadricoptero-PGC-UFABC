import gym_pybullet_drones
import gymnasium as gym
import random
import numpy as np
import pybullet as p


initial_rpys=np.array([[0.0, 0.2, 0.0]])
initial_xyzs=np.array([[0.0, 0.0, 5.0]])
randomness = random.Random(42)
env = gym.make("GPS-Distance-v0", rng=randomness, initial_rpys=initial_rpys, initial_xyzs=initial_xyzs)
print(type(env))
print("Action Space:")
print(env.action_space)  # Discrete(4) <class 'gymnasium.spaces.discrete.Discrete'>
print("Observation Space:")
print(env.observation_space)
observation, info = env.reset(seed=42)
terminated = False
i = 0
action = np.array([[0., 0., 0., 0.]])
observation, reward, terminated, truncated, info = env.step(action)
print("Action:", action)
print("Step 00, Observation:", observation[0][0][0])
print("Reward:", reward)


for _ in range(60):
   i += 1
   # action = env.action_space.sample()  # this is where you would insert your policy
   # print("Action", action)
   observation, reward, terminated, truncated, info = env.step(action)
   if i == 30:
      print("Step 30, Observation:", observation[0][0][0])
      print("Reward:", reward)
      quat = observation[1][0,:]
      # print(quat)
      # quaternion = p.getEulerFromQuaternion(quat)
      # print(quaternion)
   # print("Observation:", observation[0][0][0])
   # print(observation)
   # print("Reward:", reward)
   # print("Terminated:", terminated)
   # print("Truncated:", truncated)
   # print("Info:", info)

   # print("------------")

   # if terminated or truncated:

   #    observation, info = env.reset()
print("Step 60, Observation:", observation[0][0][0])
print("Reward:", reward)

print(i)
env.close()