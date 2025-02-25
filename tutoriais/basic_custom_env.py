import gym_pybullet_drones
import gymnasium as gym
import random
import numpy as np

initial_rpys=np.array([[0.0, 0.0, 1.0]])
env = gym.make("GPS-Distance-v0", rng=random.Random(42), initial_rpys=initial_rpys)
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
print("Step   0, Observation:", observation[0][0][0])

for _ in range(480):
   i += 1
   # action = env.action_space.sample()  # this is where you would insert your policy
   # print("Action", action)
   observation, reward, terminated, truncated, info = env.step(action)
   state = env._getDroneStateVector()
   print(state)
   if i == 240:
      print("Step 240, Velocidade:", observation[0][0][1])
   # print("Observation:", observation[0][0][0])
   # print(observation)
   # print("Reward:", reward)
   # print("Terminated:", terminated)
   # print("Truncated:", truncated)
   # print("Info:", info)

   # print("------------")

   if terminated or truncated:
      observation, info = env.reset()
print("Step 480 Observation:", observation[0][0][0])

print(i)
env.close()